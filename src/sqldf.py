import typeguard
import abc
import re
import itertools
import sqlparse
from typing import List, Dict, Optional, Union, Set
import pypama
from sqlparser import remove, white, T, I, K
from sqlparser import (
    Query, BaseExpression, Joins, AliasedFrom, AliasedExpression,
    AliasedExpressionList, Select, ExpressionList, Function, ColumnName,
    Comparison, Wildcard, Operation, Literal, Expression, BooleanOperator,
    JoinClause, Table
)
from sqlparser import parse_query


CONVERT_OP = {
    '=': '==',
    'AND': '&',
    'OR': '|',
    'NOT': '~'
}


def create_unique(prefix, namespace):
    if not prefix in namespace:
        result = prefix
    else:
        idx = max(
            (
                int(m.group(1))
                for m in (
                    re.match(f'{prefix}(\\d*)', s) for s in namespace if s
                ) if m), default=0)
        result = f"{prefix}{idx+1}"
    namespace.add(result)
    return result


class Ref:
    def __init__(self, lit):
        self.idx = lit.value


def make_coalesce(df, p):
    match p:
        case [ColumnName() as s, Literal() as q]:
            return f"{df.render(s)}.fillna({df.render(q)})"
        case _:
            return f"{df.render(p[0])}.combine_first({','.join(df.render(i) for i in p[1:])})"


# TODO: add list and other non predefined function (eg list)
AGGR_FUNCTIONS = {
    'sum': ("'sum'", 'sum'),
    'count': ("'count'", 'count'),
    'avg': ("'mean'", 'mean'),
    'list': ("list", 'list')
}

CONV_FUNCTION = {
    'round': lambda df, p: f"np.round({','.join(df.render(i) for i in p)})",
    'coalesce': make_coalesce
}


class Code:
    def __init__(self):
        self.code = []

    def __lshift__(self, value):
        self.code.append(value)
        return self

    def __str__(self):
        return ''.join(self.code)

    def __repr__(self):
        return repr(self.code)


class ExpressionAndAlias(BaseExpression):
    def __init__(self, alias: Optional[str], expression: Optional[BaseExpression]):
        self.alias = alias
        self.expression = expression

    @property
    def name(self):
        return self.alias

    def get_exposed_name(self):
        return self.alias or self.expression.name

    def __iter__(self):
        yield self.alias
        yield self.expression

    def __repr__(self):
        return f"<EA {self.alias}, {self.expression}>"

    def __str__(self):
        return f"{self.alias} AS {self.expression}"


JOIN_TYPES = {
    'join': 'inner',
    'inner': 'inner',
    'left join': 'left',
    'right join': 'right',
    'cross join': 'cross'
}


class AbstractDataFrame:
    def __init__(self, frame_name: str):
        self.frame_name = frame_name

    def get_frame_variable(self):
        return self.frame_name

    def render_column_name(self, col: ColumnName, frame_variable_name: Optional[str] = None):
        pass

    def render_function(self, name: str, parameters: List[BaseExpression]):
        return CONV_FUNCTION.get(name, name)(self, parameters)

    def render(self, expression: BaseExpression) -> str:
        # namespace = namespace or self.name
        match expression:
            case Operation(left=left, right=right, value=op):
                l = self.render(left)
                r = self.render(right)
                o = CONVERT_OP.get(op, op)
                return f"{l} {o} {r}"
            case ColumnName():
                return self.render_deref_column(expression)
            case Function(name=name, parameters=parameters):
                return self.render_function(name, parameters)
            case AliasedExpression(alias=alias, expression=expr):
                return self.render(expr)
            case Literal():
                return str(expression)
            case ExpressionAndAlias(alias=alias, expression=expr):
                return self.render(expr)
            case Expression(items=items):
                result = []
                for item in items:
                    match item:
                        case BooleanOperator(value=value):
                            result.append(CONVERT_OP.get(value.upper(), value))
                        case _ if item is not None:
                            result.extend(['(', self.render(item), ')'])
                return ' '.join(result)
            case [token]:
                return self.render(token)
            case _:
                raise ValueError()


class JoinDataFrame(AbstractDataFrame):
    def __init__(
            self,
            variable_name: str,
            df1: AbstractDataFrame,
            df2: AbstractDataFrame,
            join_clause: Optional[JoinClause],
            join_type: str
    ):
        join_type = join_type or 'join'
        self.df1 = df1
        self.df2 = df2
        self.join_clause = join_clause
        self.variable_name = variable_name
        self.jointype = JOIN_TYPES[join_type.lower()]

    def get_frame_variable(self):
        return self.variable_name

    def generate_cross_join(self, code):
        code << "pd.merge(" << self.df1.get_frame_variable(
        ) << '.assign(__crossjoin=0),'
        code << self.df2.get_frame_variable() << '.assign(__crossjoin=0),'
        code << f'left_on="__crossjoin",'
        code << f'right_on="__crossjoin",'
        code << f'suffixes=("_{self.df1.get_frame_variable()}","_{self.df2.get_frame_variable()}"),',
        code << ')' << NL

    def generate_code_body(self, code):
        if self.jointype == 'cross':
            return self.generate_cross_join(code)
        else:
            assert self.join_clause is not None, ValueError('no join clause')
            code << "pd.merge(" << self.df1.get_frame_variable() << ","
            code << self.df2.get_frame_variable() << ','
            code << f'left_on="{self.join_clause.left.name}",'
            code << f'right_on="{self.join_clause.right.name}",'
            code << f'suffixes=("_{self.df1.get_frame_variable()}","_{self.df2.get_frame_variable()}"),',
            code << f'how="{self.jointype}"'
            code << ')' << NL

    def render_column_name(self, col):
        return f"{col.name}_{col.prefix}"


class SimpleDataFrame(AbstractDataFrame):
    def get_from_(self):
        return self

    def render_column_name(self, col):
        return col.name

    def render_deref_column(self, column_name: ColumnName):
        return f"{self.get_frame_variable()}['{self.render_column_name(column_name)}']"

    def get_alias(self):
        return self.frame_name


NL = '\n'
INDENT = '    '


class DataFrame(AbstractDataFrame):
    select: List[ExpressionAndAlias]
    variable_name: str
    materialized: List[ExpressionAndAlias]
    groupby: List[ColumnName]
    select: List[ExpressionAndAlias]
    aggr: Dict[str, List[str]]
    where: Optional[str]
    namespace: Set[str]  # list of known columns
    from_: Optional[AbstractDataFrame]

    def __init__(self, name):
        self.variable_name = name
        self.from_: AbstractDataFrame = None
        self.where = None
        self.materialized = []
        self.groupby = []
        self.orderby = []
        self.limit = None
        self.array_joins = []
        self.renames = {}
        self.select = []
        self.aggr = {}
        self.namespace = {name} if name else {}

    def set_from(self, dataframe: AbstractDataFrame):
        self.from_ = dataframe

    def get_frame_variable(self):
        return self.variable_name

    def render_deref_column(self, column_name: ColumnName):
        return f"{self.from_.get_frame_variable()}['{self.from_.render_column_name(column_name)}']"

    def render_column_name(self, col):
        return col.name

    def add_where(self, where):
        if where:
            self.where = where

    def add_array_joins(self, array_joins: Optional[List[BaseExpression]]):
        if array_joins:
            self.array_joins = []
            for aj in array_joins:
                assert isinstance(aj, ColumnName)
                self.array_joins.append(aj)

    def create_name(self, expr: BaseExpression) -> str:
        r = create_unique(str(expr), self.namespace)
        r = r.replace(' ', '')
        r = re.sub('[^a-zA-Z0-9]+', '_', r)
        return '_'+r

    def materialize(self, expr: BaseExpression) -> Union[ColumnName, ExpressionAndAlias]:
        if isinstance(expr, ColumnName):
            return expr
        existing = next(
            (a for a in self.materialized if a.expression == expr), None)
        if existing:
            return existing
        else:
            col = self.create_name(expr)
            self.materialized.append(ea := ExpressionAndAlias(col, expr))
            return ea

    def set_alias(self, colname, alias):
        if colname != alias and alias is not None:
            self.rename(colname, alias)

    def rename(self, old, new):
        if new is None:
            raise ValueError()
        self.renames[old] = new

    def add_select(self, select: Select):
        for alias, expression in select:
            match expression:
                case Function(name=func_name, parameters=[parameter]) if func_name.lower() in AGGR_FUNCTIONS:
                    c = self.materialize(parameter)
                    self.aggr.setdefault(c.name, []).append(
                        AGGR_FUNCTIONS[func_name.lower()][0])
                    aggr_name = f'{c.name}_{AGGR_FUNCTIONS[func_name.lower()][1]}'
                    self.select.append(ExpressionAndAlias(aggr_name, None))
                    self.rename(
                        aggr_name, alias or self.create_name(expression))
                case ColumnName() as col:
                    self.select.append(ExpressionAndAlias(None, col))
                    if alias is not None:
                        self.rename(col.name, alias)
                case Function() | Operation() | Comparison():
                    c = self.materialize(expression)
                    self.select.append(c)
                    if alias is not None:
                        self.rename(c.name, alias)
                case Wildcard():
                    self.select.append(ExpressionAndAlias(None, expression))
                case _:
                    raise ValueError()

    def add_groupby(self, groupby: Optional[ExpressionList]):
        assert not self.groupby
        for expression in (groupby or []):
            match expression:
                case ColumnName():
                    self.groupby.append(expression)
                case Literal():
                    self.groupby.append(self.select[int(expression.value) - 1])
                case _:
                    self.groupby.append(self.materialize(expression).name)

    def add_orderby(self, orderby):
        for expression in (orderby or []):
            match expression:
                case ColumnName():
                    self.orderby.append(expression)
                case Literal():
                    self.orderby.append(self.select[int(expression.value) - 1])
                case _:
                    self.orderby.append(self.materialize(expression).name)

    def add_limit(self, limit):
        self.limit = limit

    def render_name(self, name):
        match name:
            case ExpressionAndAlias():
                return name.get_exposed_name()
            case ColumnName(name=n):
                return n
            case _:
                return str(name)

    def generate_code_body(self, code) -> Code:
        code = code or Code()
        code << '(' << self.get_frame_variable() << NL
        if self.materialized:
            code << '.assign('
            for col, exp in self.materialized:
                code << col << "= lambda _df:("
                code << SimpleDataFrame('_df').render(
                    exp) << ')' << NL << INDENT
            code << ")" << NL << INDENT
        if self.array_joins:
            for aj in self.array_joins:
                code << ".explode('" << self.render_column_name(
                    aj) << "')" << NL << INDENT
        if self.where:
            code << f'.pipe(lambda _df: _df[{self.render(self.where)}])' << NL << INDENT
        if self.groupby:
            code << ".groupby("
            code << ','.join(f"'{self.render_name(i)}'" for i in self.groupby)
            code << ',as_index=False)' << NL << INDENT
        if self.aggr:
            aggr = ','.join(
                f"'{k}':[{','.join(map(str,v))}]" for k, v in self.aggr.items())
            code << ".aggregate({" << aggr << '})' << NL << INDENT
            if not self.groupby:
                # pandas return a table column/func instead of a multi-index. Sort that:
                code << (
                    ".pipe(lambda ag: pd.DataFrame("
                    "data=ag.values.reshape((1,-1)), "
                    "columns=pd.MultiIndex.from_product([ag.columns, ag.index]))"
                    ".dropna(axis='columns'))"
                ) << NL << INDENT
            code << ".pipe(lambda df: df.set_axis(['_'.join(i for i in x if i) for x in df.columns], axis='columns'))"
            code << NL << INDENT
        if self.select and not (len(self.select) == 1 and self.select[0] == Wildcard()):
            code << '.pipe(lambda _df: _df[['
            for ae in self.select:
                if isinstance(ae.expression, Wildcard):
                    code << ']+ list(_df.columns) + ['
                else:
                    code << f"'{self.render_name(ae.get_exposed_name())}'" << ','
            code << ']])' << NL << INDENT
        if self.renames:
            code << f'.rename({self.renames}, axis="columns")' << NL << INDENT
        if self.orderby:
            code << ".sort_values("
            code << ','.join(f"'{self.render_name(i)}'" for i in self.orderby)
            code << ')' << NL << INDENT
        if self.limit:
            code << f".iloc[:{self.render(self.limit)}]"
        code << ')'
        return code


def make_dataframe(query: Query, from_: AbstractDataFrame):
    # alias = exposed_name or af.get_alias() or query.get_frame_variable()
    if isinstance(query, Query):
        # (query.get_frame_variable())
        df = DataFrame(from_.get_frame_variable())
        df.set_from(from_)
        df.add_select(query.select)
        df.add_groupby(query.groupby)
        df.add_orderby(query.orderby)
        df.add_limit(query.limit)
        df.add_where(query.where)
        df.add_array_joins(query.array_joins)
    else:
        df = query
    return df


def _code_gen(pq: Query, code: Code, exposed_name, namespace) -> AbstractDataFrame:

    for af in pq.sub_queries():
        _code_gen(af.get_from_(), code, af.get_exposed_name(), namespace)
    af = SimpleDataFrame(pq.from_[0].get_aliased_from().get_exposed_name())
    frame = make_dataframe(af.get_from_(), af.get_alias())
    for i, join in enumerate(pq.from_[1:]):  # create the joins
        # proxy for the frame that was already created
        af = SimpleDataFrame(join.get_aliased_from().get_exposed_name())
        f = make_dataframe(af.get_from_(), af.get_alias())
        join_frame_variable = create_unique('join_df', namespace)
        frame = JoinDataFrame(join_frame_variable, frame,
                              f, join.on_, join.join_type)
        code << NL << join_frame_variable << '='
        frame.generate_code_body(code)
    code << NL << exposed_name << '='
    make_dataframe(pq, frame).generate_code_body(code)
    return code


def code_gen(query: str, code: Code, exposed_name: str):
    namespace = set([exposed_name])
    pq = parse_query(query)
    return _code_gen(pq, code, exposed_name, namespace)


for name, fun in list(globals().items()):
    if isinstance(fun, type) or callable(fun) and not isinstance(fun, pypama.Pattern):
        fun = typeguard.typechecked(fun)
