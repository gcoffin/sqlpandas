import typeguard
import re
import collections
import sqlparse
import pypama
import itertools
from dataclasses import dataclass

from typing import List, Optional, Any, Callable, Union, Iterator

import re
sqlparse.keywords.SQL_REGEX.append(
    (re.compile(r'\$[a-z_][a-z0-9_]*', sqlparse.keywords.FLAGS).match, sqlparse.tokens.Name.Variable))

F = pypama.F


class TClass(pypama.F):
    """matchers a token name"""

    def __init__(self, label: str = '??', obj: type = None):
        self.obj = obj or sqlparse.tokens.Token
        pypama.F.__init__(self, self.check, name=label)

    def check(self, token):
        return token.ttype in self.obj

    def __getattr__(self, ttype):
        return TClass(ttype, getattr(self.obj, ttype))

    def clone(self):
        return self.__class__(self.__name__, self.obj)


class IClass:
    """Matches an instance of a class"""

    def __getattr__(self, clsname):
        cls = getattr(sqlparse.sql, clsname)
        return pypama.F(lambda token: isinstance(token, cls), name=clsname)


class KClass:
    """matches a keyword"""

    def check(self, name, token):
        return token.match(sqlparse.tokens.Keyword, name)

    def __getitem__(self, name):
        return pypama.F((lambda token: self.check(name, token)), name)

    __getattr__ = __getitem__


T = TClass()
I = IClass()
K = KClass()


white = T.Text.Whitespace


def remove(tokens: List[sqlparse.sql.Token], exclude: Callable = white):
    return [i for i in tokens if not exclude(i)]


class QueryComponent:
    pass


class Table(QueryComponent):
    name: str

    def __init__(self, name: str):
        self.name = name

    def get_name(self):
        return self.name

    def get_alias(self):
        return self.name

    def __repr__(self):
        return f"<Table {self.name}>"

    def __str__(self):
        return self.name


class AliasedFrom(QueryComponent):
    alias: Optional[str]
    inner_from: Union[Table, "Query"]

    def __init__(self, alias, from_):
        self.alias = alias
        self.inner_from = from_

    def get_from_(self):
        return self.inner_from

    def get_alias(self):
        return self.alias

    def get_exposed_name(self):
        if self.alias:
            return self.alias
        if isinstance(self.inner_from, Table):
            return self.inner_from.get_name()
        raise SQLSyntaxError("embedded query needs an alias")

    def __repr__(self):
        return f"<AF {repr(self.inner_from)} AS {self.alias}>"

    def __str__(self):
        if self.alias is None:
            return str(self.inner_from)
        else:
            return f'( {self.inner_from} ) AS {self.alias}'


class SQLSyntaxError(Exception):
    def __init__(self, message, token, tokens=None):
        self.message = message
        self.token = token

    def __str__(self):
        return f"Error on {self.token}: {self.message}"


def read_table(token: sqlparse.sql.Token) -> Union['Query', Table]:
    match token:
        case sqlparse.sql.Parenthesis(tokens=seq):
            seq = remove(seq, white)
            return read_query(seq[1:-1])
        case sqlparse.sql.Token() if T.Name(token):
            return Table(token.value)
        case _:
            raise SQLSyntaxError("Expected Name or ()", token)


def read_cte(token: sqlparse.sql.Token) -> AliasedFrom:
    assert I.Identifier(token)
    match remove(token.tokens):
        case [sqlparse.sql.Token(ttype=sqlparse.tokens.Token.Name, value=name),
              sqlparse.sql.Token(
                  ttype=sqlparse.tokens.Token.Keyword, value='as'),
              sqlparse.sql.Parenthesis() as par]:
            return AliasedFrom(name, read_table(par))
        case _:
            raise SQLSyntaxError(f"Expected Name or embedded query", token)


def read_from(token: sqlparse.sql.Token) -> AliasedFrom:
    match token:
        case sqlparse.sql.Identifier():
            alias, t = unpack_identifier(token)
            table = read_from(t)
            if alias is None:
                return table
            return AliasedFrom(alias, table.get_from_())
        case sqlparse.sql.Token() if T.Name(token):
            res = read_table(token)
            return AliasedFrom(res.name, res)
        case sqlparse.sql.Parenthesis(tokens=tokens):
            tokens = remove(tokens)[1:-1]
            if len(tokens) == 1:
                return read_from(tokens[0])
            else:
                res = read_table(token)
                return AliasedFrom(res.get_name(), res)
        case _:
            raise SQLSyntaxError(f"Expected Name or embedded query", token)


class BaseExpression(QueryComponent):

    def accept(self, visitor):
        getattr(visitor, 'visit' + self.__class__.__name__)


class Expression(BaseExpression):
    def __init__(self, items: List[Optional[BaseExpression] | 'BooleanOperator']):
        if isinstance(items, list):
            self.items = items
        else:
            self.items = [items]

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    def __str__(self):
        result = StrBuilder()
        for i in self.items:
            match i:
                case BooleanOperator():
                    result << i
                case x if x is not None:
                    result << "(" << i << ")"
        return str(result)
        # return ' '.join(str(i) for i in self.tokens)


@dataclass
class Function(BaseExpression):
    name: str
    parameters: List[BaseExpression]

    def __str__(self):
        return f"{self.name}({','.join(str(i) for i in self.parameters)})"


@dataclass
class ColumnName(BaseExpression):
    name: str
    prefix: str = None

    def __repr__(self):
        return f"<ColumnName {self.prefix or ''} {repr(self.name)}>"

    def __str__(self):
        if self.prefix:
            return f"{self.prefix}.{self.name}"
        return self.name


@dataclass
class Operation(BaseExpression):
    left: BaseExpression
    right: BaseExpression
    value: str

    def __str__(self):
        return f"{self.left} {self.value} {self.right}"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"


class Comparison(Operation):
    pass


@dataclass
class JoinClause(BaseExpression):
    left: ColumnName
    right: ColumnName

    def __str__(self):
        res = StrBuilder() << self.left << '=' << self.right
        return str(res)


@dataclass
class Literal(BaseExpression):
    value: str
    ttype: sqlparse.tokens._TokenType

    def __str__(self):
        return repr(self.value)[1:-1]

    def __repr__(self):
        return f"<Literal {self.value}>"


@dataclass
class BooleanOperator:
    value: str

    def __str__(self):
        return self.value


BOOLEAN_OP_PATTERN = pypama.build_pattern((K.AND | K.OR | K.NOT).capture())


def read_parameter_list(token: sqlparse.sql.Identifier) -> List[BaseExpression]:
    if token is None:
        return None
    result = []
    for identifier in token.get_identifiers():
        match identifier:
            case sqlparse.sql.Identifier():
                r = read_aliased_expression(identifier)
                assert r.get_alias() is None
                result.append(r.get_expression())
            case sqlparse.sql.Function() | sqlparse.sql.Operation() | sqlparse.sql.Comparison():
                result.append(read_expression([identifier]))
            case sqlparse.sql.Token() if T.String.Single(identifier) or T.Literal(identifier):
                result.append(read_expression([identifier]))
            case _:
                raise SQLSyntaxError(
                    "Expected identifier or expression", identifier)
    return result


def get_function_parameters(tokens):
    match tokens:
        case [
            sqlparse.sql.Token() as p1,
            sqlparse.sql.IdentifierList() as il,
            sqlparse.sql.Token() as p2
        ] if T.Punctuation(p1) and T.Punctuation(p2):
            return read_parameter_list(il)
        case [
            sqlparse.sql.Token() as p1,
            (sqlparse.sql.Identifier() | sqlparse.sql.Operation() | sqlparse.sql.Function()) as expr,
            sqlparse.sql.Token() as p2
        ] if T.Punctuation(p1) and T.Punctuation(p2):
            return [read_expression([expr])]
    raise ValueError(tokens)


def read_expression(expr: List[sqlparse.sql.Token], cls: type = Expression) -> BaseExpression:
    match expr:
        case [sqlparse.sql.Function() as fun]:
            return Function(fun.get_real_name(), get_function_parameters(remove(fun.tokens)[-1].tokens))
        case [sqlparse.sql.Token() as token_name] if T.Name(token_name):
            return ColumnName(token_name.value)
        case [sqlparse.sql.Token() as token_lit] if T.Literal(token_lit) or T.String(token_lit):
            return Literal(token_lit.value, token_lit.ttype)
        case [sqlparse.sql.Identifier() as identifier]:
            aliased_expression = read_aliased_expression(identifier)
            assert aliased_expression.get_alias() is None
            return aliased_expression.get_expression()
        case [sqlparse.sql.Operation()]:
            left, op, right = pypama.build_pattern(
                T.Operator.capture()).split(remove(expr[0].tokens))
            return Operation(read_expression(left), read_expression(right), op[0].value)
        case [sqlparse.sql.Parenthesis(tokens=t)]:
            return read_expression(remove(t)[1:-1])
        case [sqlparse.sql.Comparison()]:
            left, op, right = pypama.build_pattern(
                T.Operator.capture()).split(remove(expr[0].tokens))
            return Comparison(read_expression(left), read_expression(right), op[0].value)
        case _:
            assert len(expr) > 1
            s = iter(BOOLEAN_OP_PATTERN.split(expr))
            result = []
            for t, k in itertools.zip_longest(s, s):
                assert len(t) <= 1
                result.append(read_expression([t[0]]) if t else None)
                if k is None:
                    continue
                assert len(k) == 1
                result.append(BooleanOperator(k[0].value))
            return Expression(result)


def read_join_expression(expr: List[sqlparse.sql.Token]):
    match read_expression(expr):
        case Comparison(left=ColumnName() as left, right=ColumnName() as right, value='='):
            return JoinClause(left, right)
        case _:
            raise ValueError()


def unpack_identifier(identifier: sqlparse.sql.Identifier):
    match remove(identifier.tokens, white):
        case [expr, sqlparse.sql.Token() as token_as, alias] if K.AS(token_as):
            return alias.get_name(), expr
        case [sqlparse.sql.Token() as token_name] if T.Name(token_name):
            return None, token_name
        case [value]:
            return None, value
        case _:
            raise ValueError("Expected name [AS alias]")


@dataclass
class Join(QueryComponent):
    aliased_from: AliasedFrom
    on_: JoinClause = None
    join_type: Optional[str] = None

    def __init__(self, from_: AliasedFrom, on_: Optional[JoinClause], join_type: Optional[str]):
        self.aliased_from = from_
        self.on_ = on_
        self.join_type = join_type

    def __repr__(self):
        q = StrBuilder()
        r = q.bloc('<join', '>')
        self.to_builder(r)
        return repr(q)

    def __str__(self):
        res = StrBuilder()
        self.to_builder(res)
        return str(res)

    def get_aliased_from(self) -> AliasedFrom:
        return self.aliased_from

    def to_builder(self, result):
        result << self.join_type << self.aliased_from
        result << (self.on_ and StrBuilder('ON', self.on_))
        return result


class Joins(list, QueryComponent):

    def __str__(self):
        return ' '.join(str(i) for i in self)

    def __repr__(self):
        return ' '.join(map(repr, self))

    def get_name(self):
        match self:
            case [Join(AliasedFrom() as af)]:
                return af.get_exposed_name()
            case _:
                raise NotImplementedError()


_ARRAY_JOIN_PATTERN = pypama.S([
    T.Name.Builtin & F(lambda t: t.value.lower() == 'array'),
    T.Keyword & F(lambda t: 'join' in t.value.lower()),
    (I.Identifier.capture('array_join'))]
)
ARRAY_JOIN_PATTERN = pypama.Pattern.make(_ARRAY_JOIN_PATTERN)

_JOIN_TOKEN = (T.Keyword & F(lambda t: 'join' in t.value.lower()))
JOIN_TOKEN = pypama.Pattern.make(_JOIN_TOKEN.capture())

_JOIN_CLAUSE_PATTERN = pypama.S([
    _JOIN_TOKEN.capture('join_token'),
    (I.Parenthesis | I.Identifier).capture('join_with'),
    (K.ON + (I.Parenthesis | I.Comparison).capture('join_clause')).opt()
])


def read_array_joins(tokens) -> List[BaseExpression]:
    result = []
    if tokens:
        for i in ARRAY_JOIN_PATTERN.find_groupdict(remove(tokens)):
            t = i['array_join']
            result.append(read_expression(t))
    return result


def read_join(from_tokens, joins_tokens) -> Joins:
    seq = JOIN_TOKEN.split(remove(joins_tokens))
    assert not next(seq)
    results = Joins()
    it = iter([None, from_tokens, *seq])
    for join, select_on_tokens in itertools.zip_longest(it, it):
        if join is not None:
            assert (I.Token(join[-1]) and 'join' in join[-1].value.lower())
            join = ' '.join(map(str, join))
        match select_on_tokens:
            case [from_, sqlparse.sql.Token() as token_on, *on_clause] if K.ON(token_on):
                select_from = read_from(from_)
                on_clause = read_join_expression(on_clause)
                results.append(Join(select_from, on_clause, join))
            case [from_]:
                select_from = read_from(from_)
                results.append(Join(select_from, None, join))
            case _:
                raise SQLSyntaxError(
                    "Expected token [ON on clause]", select_on_tokens)
    return results


WITH_PATTERN = pypama.build_pattern(
    T.Keyword.CTE, I.Identifier.capture('query'))


def read_with_clause(tokens: List[sqlparse.sql.Token]) -> List[AliasedFrom]:
    result = []
    assert len(tokens)==1
    token, = tokens
    if I.Identifier(token):
        result.append(read_cte(token))
    elif I.IdentifierList(token):
        for t in token.get_identifiers():
            result.append(read_cte(t))
        
    return result


class AliasedExpression(QueryComponent):
    expression: BaseExpression
    alias: Optional[str]

    def __init__(self, alias: Optional[str], expression: BaseExpression):
        self.expression = expression
        self.alias = alias

    def get_alias(self):
        return self.alias

    def get_expression(self):
        return self.expression

    def __iter__(self):
        return iter([self.alias, self.expression])

    def __repr__(self):
        return f"<AE {repr(self.expression)} AS {self.alias}>"

    def __str__(self):
        if self.alias is None:
            return str(self.expression)
        return f'{self.expression} AS {self.alias}'


ALIASED_EXPRESSION_PATTERN = pypama.build_pattern(
    '(', (T.Name).capture('prefix'),
    T.Punctuation,
    ') ?',
    T.Name.capture('name') | (I.Function | I.Operation |
                              I.Comparison).capture('expr'),
    '(', K.AS,
    I.Identifier.capture('alias'), ')?'
)


def read_aliased_expression(token: sqlparse.sql.Token) -> AliasedExpression:
    assert I.Identifier(token)
    tokens = token.tokens
    m = ALIASED_EXPRESSION_PATTERN.match(remove(tokens))
    if not m:
        raise ValueError()
    d = m.groupdict()
    alias = token.get_alias()
    match d:
        case {'name': [name], 'prefix': [prefix]}:
            return AliasedExpression(alias, ColumnName(name.value, prefix.value))
        case {'name': [name]}:
            return AliasedExpression(alias, ColumnName(name.value))
        case {'expr': expr}:
            return AliasedExpression(token.get_alias(), read_expression(expr))
        case _:
            raise ValueError()


class AliasedExpressionList(QueryComponent):
    identifier_list: List[AliasedExpression]

    # list of Identifier
    def __init__(self, identifier_list: List[AliasedExpression]):
        assert all(isinstance(i, AliasedExpression) for i in identifier_list)
        self.identifier_list = identifier_list

    def __iter__(self):
        return iter(self.identifier_list)

    def __repr__(self):
        return f"<AEList {repr(self.identifier_list)}>"

    def __str__(self):
        return ', '.join(str(i) for i in self.identifier_list)


class Wildcard(BaseExpression):
    def __init__(self):
        pass

    def __repr__(self):
        return "<*>"

    def __str__(self):
        return "*"


def read_identifier_list(token: Optional[sqlparse.sql.Token]) -> Optional[AliasedExpressionList]:
    if token is None:
        return None
    result = []
    for identifier in token.get_identifiers():
        match identifier:
            case sqlparse.sql.Identifier():
                result.append(read_aliased_expression(identifier))
            case sqlparse.sql.Function() | sqlparse.sql.Operation() | sqlparse.sql.Comparison():
                result.append(AliasedExpression(
                    None, read_expression([identifier])))
            case sqlparse.sql.Token() if T.String.Single(identifier) or T.Literal(identifier):
                result.append(AliasedExpression(
                    None, read_expression([identifier])))
            case _:
                raise SQLSyntaxError(
                    "Expected identifier or expression", identifier)
    return AliasedExpressionList(result)


# class ExpressionList(QueryComponent):
#     expression_list: List[BaseExpression]

#     def __init__(self, expression_list: List[BaseExpression]):
#         self.expression_list = expression_list
#         assert all(isinstance(i, BaseExpression) for i in expression_list)

#     def __iter__(self):
#         return iter(self.expression_list)

#     def __str__(self):
#         return ', '.join(str(i) for i in self.expression_list)

#     def __repr__(self):
#         el = ', '.join(repr(i) for i in self.expression_list)
#         return f"<EL {el}>"


def read_expression_list(tokens: List[sqlparse.sql.Token]) -> List[BaseExpression]:
    if tokens is None:
        return None
    result = []
    for token in tokens:
        match token:
            case sqlparse.sql.Identifier():
                assert token.get_alias() is None
                result.append(read_expression([token]))
            case sqlparse.sql.Function() | sqlparse.sql.Operation() | sqlparse.sql.Comparison():
                result.append(read_expression([token]))
            case sqlparse.sql.Token() if T.Literal.Number.Integer(token):
                result.append(Literal(token.value, token.ttype))
            case sqlparse.sql.IdentifierList():
                for i in token.get_identifiers():
                    result.extend(read_expression_list([i]))
            case _:
                raise SQLSyntaxError(
                    "Expected identifier or expression", token)
    return result


class Select(QueryComponent):
    aelist: AliasedExpressionList

    def __init__(self, aelist: AliasedExpressionList):
        self.aelist = aelist

    def __iter__(self):
        return iter(self.aelist)

    def __repr__(self):
        return f"<Select {self.aelist}>"

    def __str__(self):
        return str(self.aelist)


def read_select(token: sqlparse.sql.Token) -> Select:
    match token:
        case None:
            raise ValueError()
        case sqlparse.sql.IdentifierList():
            return Select(read_identifier_list(token))
        case sqlparse.sql.Identifier():
            return Select(AliasedExpressionList([read_aliased_expression(token)]))
        case sqlparse.sql.Function() | sqlparse.sql.Operation():
            return Select(AliasedExpressionList([AliasedExpression(None, read_expression([token]))]))
        case sqlparse.sql.Token() if T.Wildcard(token):
            return Select(AliasedExpressionList([AliasedExpression(None, Wildcard())]))
        case _:
            raise ValueError(token)


def read_where(token: List[sqlparse.sql.Token]) -> Expression:
    if not (token):
        return None
    token, = token
    assert I.Where(token)
    match remove(token.tokens, white):
        case [sqlparse.sql.Token() as token_where, *expr] if token_where.value.lower() == 'where':
            return read_expression(expr)
        case _:
            raise ValueError(token)


QUERY_PATTERN = pypama.build_pattern(
    (T.Keyword.CTE + (I.Identifier | I.IdentifierList).capture('CTE')).opt(),
    T.Keyword.DML,
    T.Keyword.DISTINCT.opt().capture('DISTINCT'),
    (I.IdentifierList | I.Identifier | T.Wildcard |
     I.Function | I.Comparison | I.Operation).star().capture('SELECT'),
    K.FROM,
    pypama.S([
        (I.Parenthesis | I.Identifier).capture('FROM'),
        _ARRAY_JOIN_PATTERN.star().capture('ARRAYJOINS'),
        _JOIN_CLAUSE_PATTERN.star().capture('JOINS')]
    ),
    I.Where.opt().capture('WHERE'),
    (K['GROUP BY'] + (I.IdentifierList | I.Identifier | I.Function |
     I.Comparison | I.Operation | T.Literal.Number).capture('GROUPBY')).opt(),
    I.Having.opt().capture('HAVING'),
    (K['ORDER BY'] + (I.IdentifierList | I.Identifier | T.Literal.Number).capture('ORDERBY')
     ).opt().capture('ORDERBY'),
    (K.Limit + T.Literal.Number.capture('LIMIT')).opt(),
    '$',
    **globals())


class StrBuilder:
    def __init__(self, *args):
        self.buffer = [i for i in args]

    def __lshift__(self, values: Any):
        if isinstance(values, list):
            values = StrBuilder(values)
        self.buffer.append(values)
        return self

    def flatten(self):
        for i in self.buffer:
            if isinstance(i, StrBuilder):
                yield from i.flatten()
            else:
                yield i

    def bloc(self, open: str, close: str):
        res = StrBuilder()
        self << open << res << close
        return res

    def __str__(self):
        res = ' '.join(str(i) for i in self.flatten() if i is not None)
        return res

    def __repr__(self):
        return ' '.join((i if isinstance(i, str) else repr(i)) for i in self.buffer if i is not None)


class Query(QueryComponent):
    def __init__(
        self,
        select: Optional[Select],
        from_: Optional[Joins],
        array_joins: Optional[List[BaseExpression]] = None,
        distinct: bool = False,
        where: Optional[BaseExpression] = None,
        groupby: Optional[List[BaseExpression]] = None,
        orderby: Optional[List[BaseExpression]] = None,
        having: Optional[List[BaseExpression]] = None,
        limit: Optional[Literal] = None,
        with_clause: Optional[List[AliasedFrom]] = None
    ):
        self.select = select
        self.from_ = from_
        self.array_joins = array_joins
        self.distinct = distinct
        self.where = where
        self.groupby = groupby
        self.orderby = orderby
        self.having = having
        self.limit = limit
        self.with_clause = with_clause

    def sub_queries(self) -> Iterator[AliasedFrom]:
        yield from self.with_clause
        for join in self.from_:
            match join:
                case Join(aliased_from=AliasedFrom(inner_from=Table())):
                    pass
                case Join(aliased_from=AliasedFrom(inner_from=Query() as q) as af):
                    # yield from q.sub_queries()
                    yield af

    # def get_frame_name(self) -> Optional[str]:
    #     return self.from_.get_name()

    def get_name(self):
        return None

    def to_str_builder(self, builder: StrBuilder) -> StrBuilder:
        builder << 'SELECT' << self.select << 'FROM' << self.from_
        builder << (self.where and StrBuilder('WHERE', self.where))
        builder << (self.groupby and StrBuilder('GROUP BY', self.groupby))
        builder << (self.orderby and StrBuilder('ORDER BY', StrBuilder(*self.orderby)))
        builder << (self.having and StrBuilder('HAVING', self.having))
        builder << (self.limit and StrBuilder('LIMIT', self.limit))
        return builder

    def __repr__(self):
        result = StrBuilder()
        q = result.bloc('<Query', '>')
        self.to_str_builder(q)
        return repr(result)

    def __str__(self):
        result = StrBuilder()
        self.to_str_builder(result)
        return str(result)


def read_query(tokens: List[sqlparse.sql.Token]) -> Query:
    seq = remove(tokens, white)

    res = QUERY_PATTERN.match(seq)

    if not res:
        raise ValueError(seq)
    else:
        vars = res.groupdict()
        return Query(
            read_select(vars['SELECT'][0]),
            read_join(vars['FROM'], vars['JOINS']),
            read_array_joins(vars['ARRAYJOINS']),
            bool(vars.get('DISTINCT', False)),
            read_where(vars.get('WHERE')),
            read_expression_list(vars.get('GROUPBY')) or None,
            read_expression_list(vars.get('ORDERBY')) or None,
            read_where(vars.get('HAVING')),
            read_expression(vars['LIMIT']) if 'LIMIT' in vars else None,
            read_with_clause(vars['CTE']) if 'CTE' in vars else []
        )


def parse_query(s: str) -> Query:
    tokens = sqlparse.parse(s)[0].tokens
    return read_query(tokens)


for name, fun in list(globals().items()):
    if name in ["TClass", "IClass", "KClass"]:
        continue
    if isinstance(fun, (TClass, KClass, IClass)):
        continue
    if isinstance(fun, type) or callable(fun):
        fun = typeguard.typechecked(fun)
