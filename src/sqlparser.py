import typeguard
import re
import collections
import sqlparse
import pypama
import itertools
from typing import List, Optional, Any, Callable, Union

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


class Visitable:
    def accept(self, visitor):
        pass


class Table:
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


class AliasedFrom:
    alias: Optional[str]
    from_: Union[Table, "Query"]

    def __init__(self, alias, from_):
        self.alias = alias
        self.from_ = from_

    def get_from_(self):
        return self.from_

    def get_alias(self):
        return self.alias

    def __repr__(self):
        return f"<AF {repr(self.from_)} AS {self.alias}>"

    def __str__(self):
        if self.alias is None:
            return str(self.from_)
        else:
            return f'( {self.from_} ) AS {self.alias}'

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


class Expression:
    def __init__(self, tokens: List[sqlparse.sql.Token]):
        if isinstance(tokens, list):
            self.tokens = tokens
        else:
            self.tokens = [tokens]

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    def __str__(self):
        return str(StrBuilder(*self.tokens))
        # return ' '.join(str(i) for i in self.tokens)

    def adapt(self, namespace):
        pass


# def ttype_is(ttype, ttype_ancestor):
#     while ttype:
#         if ttype is ttype_ancestor:
#             return True
#         ttype = ttype.parent
#     return False


# def token_is(token, ttype):
#     assert isinstance(token, sqlparse.sql.Token)
#     return ttype_is(token.ttype, ttype)


def read_expression(expr: List[sqlparse.sql.Token], cls: type = Expression) -> Union[Expression, 'JoinExpression']:
    return cls(expr)
    # match expr:
    #     case sqlparse.sql.Token() if expr.value == 'Name':
    #             return Expression(expr)
    #     case sqlparse.sql.Token() if token_is(expr, sqlparse.tokens.Token.Literal):
    #             return Expression(expr)
    #     case sqlparse.sql.Function() | sqlparse.sql.Operation() | sqlparse.sql.Comparison():
    #         return Expression(expr)
    #     case _:
    #         raise ValueError()


class JoinExpression(Expression):
    pass


def read_join_expression(expr: sqlparse.sql.Token):
    return read_expression(expr, JoinExpression)


JOIN_PATTERN = pypama.build_pattern(
    (T.Keyword & F(lambda t: 'join' in t.value.lower())).capture()
)


def unpack_identifier(identifier: sqlparse.sql.Identifier):
    match remove(identifier.tokens, white):
        case [expr, sqlparse.sql.Token() as token_as, alias] if K.AS(token_as):
            return alias.get_name(), expr
        case [sqlparse.sql.Token() as token_name] if K.Name(token_name):
            return token_name.value, token_name
        case [value]:
            return None, value
        case _:
            raise ValueError("Expected name [AS alias]")


class Join:
    def __init__(
        self,
        from_: AliasedFrom,
        on_: JoinExpression = None,
        join: Optional[str] = None
    ):
        assert isinstance(from_, AliasedFrom)
        assert on_ is None or isinstance(on_, Expression)
        self.from_ = from_
        self.on_ = on_
        self.join = join

    def __repr__(self):
        q = StrBuilder()
        r = q.bloc('<join', '>')
        self.to_builder(r)
        return repr(q)

    def __str__(self):
        res = StrBuilder()
        self.to_builder(res)
        return str(res)

    def to_builder(self, result):
        result << self.from_
        result << (self.on_ and StrBuilder('ON', self.on_))
        result << StrBuilder(self.join)
        return result


class Joins:
    join_list: str

    def __init__(self, join_list: List[Join]):
        self.join_list = join_list

    def __str__(self):
        return ' '.join(str(i) for i in self.join_list)

    def __repr__(self):
        return ' '.join(map(repr, self.join_list))


def read_join(tokens) -> Joins:
    seq = list(JOIN_PATTERN.split(remove(tokens, white)))
    results = []
    it = iter(seq)
    for select_on_tokens, join in itertools.zip_longest(it, it):
        if join is not None:
            assert (I.Token(join[0]) and 'join' in join[0].value.lower())
            join = join[0].value
        match select_on_tokens:
            # case [sqlparse.sql.Identifier() as identifier]:
            #     # alias, from_ = unpack_identifier(identifier)
            #     select_from = read_from(identifier)
            #     results.append(Join(select_from, None, join))
            case [from_, sqlparse.sql.Token() as token_on, *on_clause] if K.ON(token_on):
                select_from = read_from(from_)
                on_clause = read_join_expression(on_clause)
                results.append(Join(select_from, on_clause, join))
            case [from_]:
                select_from = read_from(from_)
                results.append(Join(select_from, None, join))
            case _:
                raise SQLSyntaxError("Expected token [ON on clause]", select_on_tokens)
    return Joins(results)


class Column(Expression):
    name: str

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"<Column {repr(self.name)}>"

    def __str__(self):
        return self.name


class AliasedExpression:
    expression: Expression
    alias: Optional[str]

    def __init__(self, alias: Optional[str], expression: Expression):
        self.expression = expression
        self.alias = alias

    def get_alias(self):
        return self.alias

    def __repr__(self):
        return f"<AE {repr(self.expression)} AS {self.alias}>"

    def __str__(self):
        if self.alias is None:
            return str(self.expression)
        return f'{self.expression} AS {self.alias}'


def read_aliased_expression(token: sqlparse.sql.Token) -> AliasedExpression:
    assert I.Identifier(token)
    tokens = token.tokens
    match remove(tokens, white):
        case [expr, _as, alias] if K.AS(_as):
            return AliasedExpression(alias.get_name(), Expression([expr]))
        case [name] if T.Name(name):
            return AliasedExpression(name.value, Column(name.value))
        case _:
            raise SQLSyntaxError("Expected value [AS alias]", token)


class AliasedExpressionList:
    identifier_list: List[AliasedExpression]

    # list of Identifier
    def __init__(self, identifier_list: List[AliasedExpression]):
        assert all(isinstance(i, AliasedExpression) for i in identifier_list)
        self.identifier_list = identifier_list

    def __repr__(self):
        return f"<AEList {repr(self.identifier_list)}>"

    def __str__(self):
        return ', '.join(str(i) for i in self.identifier_list)


class Wildcard(AliasedExpressionList):
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
                result.append(Expression(identifier))
            case _:
                raise SQLSyntaxError("Expected identifier or expression")
    return AliasedExpressionList(result)


class ExpressionList:
    expression_list: List[Expression]

    def __init__(self, expression_list: List[Expression]):
        self.expression_list = expression_list
        assert all(isinstance(i, Expression) for i in expression_list)

    def __str__(self):
        return ', '.join(str(i) for i in self.expression_list)

    def __repr__(self):
        el = ', '.join(repr(i) for i in self.expression_list)
        return f"<EL {el}>"


def read_expression_list(tokens: List[sqlparse.sql.Token]) -> ExpressionList:
    if tokens is None:
        return None
    result = []
    for token in tokens:
        match token:
            case sqlparse.sql.Identifier():
                assert token.get_alias() is None
                result.append(Expression([token]))
            case sqlparse.sql.Function() | sqlparse.sql.Operation() | sqlparse.sql.Comparison():
                result.append(Expression([token]))
            case _:
                raise SQLSyntaxError("Expected identifier or expression", token)
    if not result:
        return None
    else:
        return ExpressionList(result)


class Select:
    aelist: AliasedExpressionList

    def __init__(self, aelist: AliasedExpressionList):
        self.aelist = aelist

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
        # case sqlparse.sql.Function() | sqlparse.sql.Operation():
        #     return {None: token}
        case sqlparse.sql.Token() if T.Wildcard(token):
            return Select(Wildcard())
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
    T.Keyword.DML,
    T.Keyword.DISTINCT.opt().capture('DISTINCT'),
    (I.IdentifierList | I.Identifier | T.Wildcard).capture('SELECT'),
    K.FROM,
    (
        (I.Parenthesis | I.Identifier)
        + (
            (T.Keyword & F(lambda t: 'join' in t.value.lower()))
            + (I.Parenthesis | I.Identifier)
            + (K.ON + (I.Parenthesis | I.Comparison)).opt()
        ).star()
    ).capture('FROM'),
    I.Where.opt().capture('WHERE'),
    (K['GROUP BY'] + (I.IdentifierList | I.Identifier).capture('GROUPBY')).opt(),
    I.Having.opt().capture('HAVING'),
    (K['ORDER BY'] + (I.IdentifierList | I.Identifier).capture('ORDERBY')
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
        return ' '.join(str(i) for i in self.flatten() if i is not None)

    def __repr__(self):
        return ' '.join((i if isinstance(i, str) else repr(i)) for i in self.buffer if i is not None)


class Query:
    def __init__(self,  select, from_, distinct=False, where=None, groupby=None, orderby=None, having=None, limit=None):
        self.select = select
        self.from_ = from_
        self.distinct = distinct
        self.where = where
        self.groupby = groupby
        self.orderby = orderby
        self.having = having
        self.limit = limit

    def get_name(self) -> Optional[str]:
        return None

    def to_str_builder(self, builder: StrBuilder) -> StrBuilder:
        builder << 'SELECT' << self.select << 'FROM' << self.from_
        builder << (self.where and StrBuilder('WHERE', self.where))
        builder << (self.groupby and StrBuilder('GROUP BY', self.groupby))
        builder << (self.orderby and StrBuilder('ORDER BY', self.orderby))
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
            read_join(vars['FROM']),
            bool(vars.get('DISTINCT', False)),
            read_where(vars.get('WHERE')),
            read_expression_list(vars.get('GROUPBY')),
            read_expression_list(vars.get('ORDERBY')),
            read_where(vars.get('HAVING')),
            read_expression(vars['LIMIT']) if 'LIMIT' in vars else None
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


# FROM_PATTERN = pypama.build_pattern(
#         I.Identifier.capture('identifier') |
#         (T.Keyword.DML + pypama.ANY.star()).capture('select') | #should be SELECT
#         T.Name.capture('table') |
#         (I.Parenthesis.capture('parenthesis'))
#         # (K.AS + I.Identifier.capture('alias')).opt()
#     )


# def merge_dicts(dctlist):
#     idx = 0
#     res = {}
#     for d in dctlist:
#         for k,v in d.items():
#             if k is None:
#                 k = f"_col{idx}"
#                 idx += 1
#             assert not k in res
#             res[k] = v
#     return res


# class Item:
#     def __init__(self, **dct):
#         self.data = dct

#     def __getattr__(self, name):
#         return self.data.get(name)

#     def __repr__(self):
#         return f"<{self.__class__.__name__} {self.data}>"

#     @classmethod
#     def make(cls, tokens):
#         seq = remove(tokens, white)
#         res = cls.pattern.match(seq)
#         if res:
#             return cls(**res.groupdict())
#         else:
#             print('ERROR')
#             for i in seq:print(i)
#             raise ValueError(res)
