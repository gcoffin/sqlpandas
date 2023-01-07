import sqlparser

INDENT = '  '
import re

def prettify(obj):
    s = repr(obj)
    indent = 0
    res = []
    for o in re.split(' |(>)',s):
        if o is None: continue
        if o.startswith('<'):
            res.append('\n')
            res.append(INDENT*indent)
            indent += 1
        if o == '>':
            indent -= 1
            res.append('\n')
            res.append(INDENT*indent)
            res.append('>')
        else:
            res.append(o)
    return ' '.join(res).strip()


def assert_same(query):
    pq = sqlparser.parse_query(query)
    actual = str(pq)
    # print(prettify(pq))
    assert query.strip() == actual
    
def test_idempotent():
    assert_same("""SELECT a AS a FROM ( b ) AS b""")
    assert_same("""SELECT a AS a FROM ( b ) AS b ORDER BY a""")
    assert_same("""SELECT a1 AS a1 FROM ( SELECT a AS a FROM ( b ) AS b ) AS d LIMIT 10""")
    assert_same("""SELECT a AS a FROM ( b ) AS b WHERE c='toto' AND d<=10""")
    assert_same("""SELECT a AS a FROM ( b ) AS b JOIN ( c ) AS c""")
    assert_same("""SELECT a AS a FROM ( b ) AS b JOIN ( c ) AS c ON b.a = c.a""")
    assert_same("""SELECT a AS a FROM ( b ) AS b JOIN ( c ) AS cc ON b.a = c.a OUTER JOIN ( d ) AS d ON c.a=d.b""")
    assert_same('SELECT a AS a FROM ( b ) AS b LEFT JOIN ( SELECT b AS b FROM ( d ) AS d ) AS cc ON b.a = cc.a')

def test_simple():
    q = """select a from b"""
    pq = sqlparser.parse_query(q)
    assert pq.from_ is not None
    assert str(pq) == "SELECT a AS a FROM ( b ) AS b"

def test_simple_where():
    q = """select a1, a2 as b from b where c='TOTO'"""
    pq = sqlparser.parse_query(q)
    assert pq.from_ is not None
    # assert str(pq) == "SELECT a1 AS a1, a2 AS b from ( b ) AS b WHERE c='TOTO'"
    assert repr(pq) == ("<Query SELECT <Select a1 AS a1, a2 AS b> FROM <join <AF <Table b> AS b>  > "
    "WHERE <Expression c='TOTO'> >")

def test_from_query():
    q = """select a1 from (select a from b) as d limit 10"""
    pq = sqlparser.parse_query(q)
    assert pq.from_ is not None
    assert repr(pq) == ("<Query SELECT <Select a1 AS a1> "
    "FROM <join <AF <Query SELECT <Select a AS a> FROM <join <AF <Table b> AS b>  > > AS d>  >"
    " LIMIT <Expression 10> >")
    

if __name__ == "__main__":
    test_idempotent()
    test_simple()
    test_simple_where()
    # test_from_query()
    # test_from_par()
    # test_from_join()
    # test_from_join_on()