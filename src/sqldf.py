import re
from typing import List, Dict, Optional, Set
import pypama
from sqlparser import remove, white, T, I, K
from sqlparser import (
    Query, BaseExpression,
    Select, Function, ColumnName,
    Comparison, Wildcard, Operation, Literal, Expression, BooleanOperator,
    JoinClause, Distinct, ValueList
)
from sqlparser import parse_query


def make_binary_op(v):
    return f'{{l}} {v} {{r}}'


_BINARY_OPS = {
    '=': '==',
    'AND': '&',
    'OR': '|',
    '<>': '!=',
    **{k: k for k in ['<=', '>=', *'+/*-%<>']}
}

CONVERT_OP = {
    'NOT': '~{r}',
    'IN': '{l}.isin({r})',
    'LIKE': "{l}.str.match(({r}).replace('%','.*'))",
    'REGEXP': "{l}.str.match({r})",
    ** {k: make_binary_op(v) for k, v in _BINARY_OPS.items()}
}


def create_unique(prefix: str, namespace: Set[str]):
    if not prefix in namespace:
        result = prefix
    else:
        idx = max((int(m.group(1))
                   for m in (
            re.match(f'{prefix}(\\d*)', s) for s in namespace if s
        ) if m), default=0)
        result = f"{prefix}{idx+1}"
    namespace.add(result)
    return result


def make_coalesce(df: 'AbstractDataFrame', p: List[BaseExpression]):
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
    'list': ("list", 'list'),
    'set': ("lambda t: frozenset(t.unique())", '<lambda>'),
    'min': ("'min'", 'min'),
    'max': ("'max'", 'max'),
}

CONV_FUNCTION = {
    'round': lambda df, p: f"np.round({','.join(df.render(i) for i in p)})",
    'coalesce': make_coalesce,
    'log': "np.log",
    'pow': ".pow",
    'sin': 'np.sin',
    'cos': 'np.cos',
    'tan': 'np.tan',
    'lower': '.str.lower',
    'upper': '.str.upper',
    'length': '.str.len',
    'slice': '.str.slice'
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

    def __iter__(self):
        yield self.alias
        yield self.expression

    def __repr__(self):
        return f"<EA {self.alias}, {self.expression}>"


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

    def render_function(self, name: str, parameters: List[BaseExpression] | Distinct):
        fun = CONV_FUNCTION.get(name, name)
        is_distinct = False
        if callable(fun):
            return fun(self, parameters)
        else:
            if fun.startswith('.'):
                this, *params = parameters
                return f"{self.render(this)}{fun}({','.join(map(self.render, params))})"
            else:
                return f"{fun}({','.join(map(self.render, parameters))})"

    def render(self, expression: BaseExpression) -> str:
        match expression:
            case Operation(left=left, right=right, value=op):
                l = self.render(left)
                r = self.render(right)
                return CONVERT_OP[op.upper()].format(l=l, r=r)
            case ColumnName():
                return self.render_deref_column(expression)
            case Function(name=name, parameters=parameters):
                return self.render_function(name, parameters)
            case Literal():
                return str(expression)
            case Expression(items=items):
                result = []
                for item in items:
                    match item:
                        case BooleanOperator(value=value):
                            result.append(
                                CONVERT_OP[value.upper()].format(r='', l=''))
                        case _ if item is not None:
                            result.extend(['(', self.render(item), ')'])
                return ' '.join(result)
            case ValueList(values=values):
                return f"({','.join(self.render(i) for i in values)})"
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

    def _generate_cross_join(self, code: Code):
        code << "pd.merge(" << self.df1.get_frame_variable(
        ) << '.assign(__crossjoin=0),'
        code << self.df2.get_frame_variable() << '.assign(__crossjoin=0),'
        code << f'left_on="__crossjoin",'
        code << f'right_on="__crossjoin",'
        code << f'suffixes=("_{self.df1.get_frame_variable()}","_{self.df2.get_frame_variable()}"),',
        code << ')' << NL

    def generate_code_body(self, code: Code):
        if self.jointype == 'cross':
            return self._generate_cross_join(code)
        else:
            assert self.join_clause is not None, ValueError('no join clause')
            code << "pd.merge(" << self.df1.get_frame_variable() << ","
            code << self.df2.get_frame_variable() << ','
            code << f'left_on="{self.join_clause.left.name}",'
            code << f'right_on="{self.join_clause.right.name}",'
            code << f'suffixes=("_{self.df1.get_frame_variable()}","_{self.df2.get_frame_variable()}"),',
            code << f'how="{self.jointype}"'
            code << ')' << NL

    def render_column_name(self, col: ColumnName):
        if col.prefix is None:
            return col.name
        return f"{col.name}_{col.prefix}"


class SimpleDataFrame(AbstractDataFrame):
    def get_from_(self):
        return self

    def render_column_name(self, col: ColumnName):
        return col.name

    def render_deref_column(self, column_name: ColumnName):
        return f"{self.get_frame_variable()}['{self.render_column_name(column_name)}']"

    def get_alias(self):
        return self.frame_name


NL = '\n'
INDENT = '    '


def normalize(r: str):
    r = r.replace(' ', '')
    r = re.sub('[^a-zA-Z0-9]+', '_', r)
    return '_'+r


class DataFrame(AbstractDataFrame):
    variable_name: str
    materialized: List[ExpressionAndAlias]
    groupby: List[ColumnName]
    select: List[ExpressionAndAlias]
    distinct: bool = False
    aggr: Dict[str, List[str]]
    where: Optional[str]
    namespace: Set[str]  # list of known columns
    from_: Optional[AbstractDataFrame]

    def __init__(self, name: str):
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
        self.distinct = False
        self.namespace = {name} if name else {}

    def set_from(self, dataframe: AbstractDataFrame):
        self.from_ = dataframe

    def get_frame_variable(self):
        return self.variable_name

    def render_deref_column(self, column_name: ColumnName):
        return f"{self.from_.get_frame_variable()}['{self.from_.render_column_name(column_name)}']"

    def render_column_name(self, col: ColumnName):
        return col.name

    def add_where(self, where: Optional[BaseExpression]):
        if where:
            self.where = where

    def add_array_joins(self, array_joins: Optional[List[BaseExpression]]):
        if array_joins:
            self.array_joins = []
            for aj in array_joins:
                assert isinstance(aj, ColumnName)
                self.array_joins.append(aj)

    def create_name(self, expr: BaseExpression) -> str:
        return create_unique(normalize(str(expr)), self.namespace)

    def materialize(self, expr: BaseExpression) -> ColumnName:
        if isinstance(expr, ColumnName):
            return expr
        existing = next(
            (a for a in self.materialized if a.expression == expr), None)
        if existing:
            return ColumnName(existing.alias)
        else:
            col = self.create_name(expr)
            self.materialized.append(ea := ExpressionAndAlias(col, expr))
            return ColumnName(ea.alias)

    def rename(self, old: str, new: str):
        if new is None:
            raise ValueError()
        self.renames[old] = new

    def set_distinct(self, distinct):
        self.distinct = distinct

    def add_select(self, select: Select):
        for alias, expression in select:
            match expression:
                case Function(name=func_name, parameters=Distinct(value=[parameter])) if func_name.lower() in AGGR_FUNCTIONS:
                    c = self.materialize(parameter)
                    assert func_name.lower() == 'count'
                    self.aggr.setdefault(c.name, []).append("'nunique'")
                    aggr_name = f'{c.name}_nunique'
                    self.select.append(ColumnName(aggr_name))
                    self.rename(
                        aggr_name, alias or self.create_name(expression))
                case Function(name=func_name, parameters=[parameter]) if func_name.lower() in AGGR_FUNCTIONS:
                    c = self.materialize(parameter)
                    self.aggr.setdefault(c.name, []).append(
                        AGGR_FUNCTIONS[func_name.lower()][0])
                    aggr_name = f'{c.name}_{AGGR_FUNCTIONS[func_name.lower()][1]}'
                    self.select.append(ColumnName(aggr_name))
                    self.rename(
                        aggr_name, alias or self.create_name(expression))
                case ColumnName() as col:
                    self.select.append(col)
                    if alias is not None:
                        self.rename(col.name, alias)
                case Function() | Operation() | Comparison():
                    c = self.materialize(expression)
                    self.select.append(c)
                    if alias is not None:
                        self.rename(c.name, alias)
                case Wildcard():
                    self.select.append(expression)
                case _:
                    raise ValueError()

    def add_groupby(self, groupby: Optional[List[BaseExpression]]):
        assert not self.groupby
        for expression in (groupby or []):
            match expression:
                case ColumnName():
                    self.groupby.append(expression)
                case Literal():
                    self.groupby.append(self.select[int(expression.value) - 1])
                case _:
                    self.groupby.append(self.materialize(expression))

    def add_orderby(self, orderby: Optional[List[BaseExpression]]):
        for expression in (orderby or []):
            match expression:
                case ColumnName():
                    self.orderby.append(expression)
                case Literal():
                    self.orderby.append(self.select[int(expression.value) - 1])
                case _:
                    self.orderby.append(self.materialize(expression).name)

    def add_limit(self, limit: Optional[BaseExpression]):
        self.limit = limit

    def render_name(self, name):
        match name:
            case ExpressionAndAlias():
                # name.get_exposed_name()
                return self.from_.render_column_name(name.expression)
            case ColumnName(name=n):
                return n
            case _:
                return str(name)

    def generate_code_body(self, code: Code) -> Code:
        code = code or Code()
        code << '(' << self.get_frame_variable() << NL
        if self.materialized:
            code << '.assign('
            for col, exp in self.materialized:
                code << col << "= lambda _df:("
                code << SimpleDataFrame('_df').render(
                    exp) << '),' << NL << INDENT
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
                )
                code << NL << INDENT
                code << ".pipe(lambda df: df.set_axis(['_'.join(i for i in x if i) for x in df.columns], axis='columns'))"
                code << ".dropna(axis='columns'))"
                code << NL << INDENT
            else:
                code << ".pipe(lambda df: df.set_axis(['_'.join(i for i in x if i) for x in df.columns], axis='columns'))"
                code << NL << INDENT
        if self.select and not (len(self.select) == 1 and isinstance(self.select[0], Wildcard)):
            code << '.pipe(lambda _df: _df[['
            for ae in self.select:
                if isinstance(ae, Wildcard):
                    code << ']+ list(_df.columns) + ['
                else:
                    code << f"'{self.from_.render_column_name(ae)}'" << ','
            code << ']]'
            if self.distinct:
                code << ".drop_duplicates()"
            code << ')' << NL << INDENT
        if self.renames:
            code << f'.rename({self.renames}, axis="columns")' << NL << INDENT
        if self.orderby:
            code << ".sort_values(["
            code << ','.join(
                f"'{self.from_.render_column_name(i)}'" for i in self.orderby)
            code << '])' << NL << INDENT
        if self.limit:
            code << f".iloc[:{self.render(self.limit)}]"
        code << ')'
        return code


def make_dataframe(query: Query | AbstractDataFrame, from_: AbstractDataFrame):
    # alias = exposed_name or af.get_alias() or query.get_frame_variable()
    if isinstance(query, Query):
        # (query.get_frame_variable())
        df = DataFrame(from_.get_frame_variable())
        df.set_from(from_)
        df.add_select(query.select)
        df.set_distinct(query.distinct)
        df.add_groupby(query.groupby)
        df.add_orderby(query.orderby)
        df.add_limit(query.limit)
        df.add_where(query.where)
        df.add_array_joins(query.array_joins)
    else:
        df = query
    return df


def _code_gen(pq: Query, code: Code, exposed_name, namespace) -> Code:
    """generate python code from a query or a sub query"""
    for af in pq.sub_queries():
        _code_gen(af.get_from_(), code, af.get_exposed_name(), namespace)
    af = SimpleDataFrame(pq.from_[0].get_aliased_from().get_exposed_name())
    frame = make_dataframe(af.get_from_(), af)
    for i, join in enumerate(pq.from_[1:]):  # create the joins
        # proxy for the frame that was already created
        af = SimpleDataFrame(join.get_aliased_from().get_exposed_name())
        f = make_dataframe(af.get_from_(), af)
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


if __name__ == "__main__":
    query = input()
    code = Code()
    code_gen(query, code, 'resdf')
    print(str(code))
