from sqldf import parse_query, Code, DataFrame, code_gen
import numpy as np
import pytest

HEADER = ['id', 'name', 'gender', 'height', 'town']
DATA = [
    [1, 'Raymond',  'M', 182, 'Paris'],
    [2, 'Josiane',   "F", 165, 'Lyon'],
    [3, 'Kevin',     'M', 190, 'Paris'],
    [4, 'Paulette',  'F', 162, 'Lyon'],
    [5, 'Farid',     'M', 184, 'Lyon']
]

HEADER2 = ['town', 'habitants']
DATA2 = [
    ['Paris', 2e6],
    ['Lyon', 1e6],
    ['Sainte-Menehould', 4.6e3]
]

INTRO1 = f"""import pandas as pd
df = pd.DataFrame(data={DATA}, columns={HEADER})
df2 = pd.DataFrame(data={DATA2}, columns={HEADER2})

import numpy as np
import sqlite3
"""

INTRO = INTRO1 + """
con = sqlite3.connect(":memory:")
cursor = con.cursor()
df.to_sql('df', con, index=False)
df2.to_sql('df2', con, index=False)
resql = pd.read_sql('''{q}''', con=con)
"""

QUERIES = [
    """select df.height, coalesce(df.name,'-') from df""",
    """select df2.town as toto, name from df join df2 on df.town=df2.town order by name""",
    """select df.id as a from df""",
    """select count(height) from df""",
    """select town, avg(height) from df where gender='M' and not name='Paulette' group by town""",
    """select count(id) from df """,
    """select count(id) as cnt, sum(id), round(height/10) as height_range from df group by round(height/10)""",
    """select height, round(height/10) as height_range from df""",
    """select b from (select height*2 as b from df) as foo""",
    """select count(id) as cnt, sum(id) as s, round(height/10) as height_range from df group by round(height/10)""",
    """select df2.habitants, df.name from df join df2 on df.town=df2.town order by name""",
    """select df.id as a from df order by a limit 3""",
    """select coalesce(id,-1) from df right join (select town as town2, habitants from df2) as df3
     on df.town=df3.town2 order by 1""",
    """select * from (select * from df) as df1""",
    """select * from (select * from (select * from df) as df1) as df2""",
    """select avg(height), round(height/10) as height_range from df group by 2""",
    """select * from (select id from df) as df1 cross join (select id from df) as df2 where df1.id < df2.id""",
]


def drop_hidden(df):
    cols = [c for c in df.columns if not c.startswith('__')]
    return df[cols]


@pytest.mark.parametrize('q', [(i) for i in QUERIES])
def test_query(q):
    code = Code()
    code_gen(q, code, 'resdf')

    d = {}
    try:
        exec(INTRO.format(q=q) + str(code), d)
    except Exception:
        print(INTRO.format(q=q) + str(code))
        raise
    print('----------')
    print(q)
    print(d['resdf'])

    if not (np.array_equal(drop_hidden(d['resdf']).values, d['resql'].values)):
        print()
        print(str(code))
        print()
        print(d['resdf'])
        print(d['resql'])
        raise ValueError()


QUERIES2 = [
    # (
    #     """select town, list(height) as height_range from df group by 1""",
    #     [['Lyon',  [165, 162, 184]], ['Paris', [182, 190]]]
    # ),
    (
        """select * from 
            (select town, list(height) as height_range from df group by 1) as dflist
            array join dflist.height_range""",
        [['Lyon',  165],
         ['Lyon',  162],
         ['Lyon',  184],
         ['Paris', 182],
         ['Paris', 190]
         ]

    )
]


@pytest.mark.parametrize('q,data', QUERIES2)
def test_query2(q, data):
    code = Code()
    code_gen(q, code, 'resdf')

    d = {}
    try:
        exec(INTRO1.format(q=q) + str(code), d)
    except Exception:
        print(INTRO1.format(q=q) + str(code))
        raise
    print('----------')
    print(q)
    print(d['resdf'])

    if not (np.array_equal(d['resdf'].values.tolist(), data)):
        print()
        print(str(code))
        print()
        print(d['resdf'])
        raise ValueError()


if __name__ == "__main__":
    if 0:
        for q in QUERIES:
            test_query(q)
    else:
        for q in QUERIES2:
            test_query2(*q)
