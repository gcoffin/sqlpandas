
# sqlpandas


A SQL to python/pandas compiler.

Presentation
------------

Pandas is a powerful data manipulation tool. On the other hand, SQL is a data query and transformation language. These is an overlap between those two.
For instance, projection (selection of 2 columns) can be expressed in SQL:
```select a, b from table``` and in pandas: ```table[['a','b']]```.

As for grouping:
```select a, sum(b) from table group by a``` is equivalent to
```table.groupby(['a'], as_index=False)[['b']].sum()```

For complex transformations, SQL is more expressive than pandas. This tool provides a transformation from SQL to pandas.

Usage
-----
```
python sqldf.py
>>> enter your query
[output is the equivalent python code with pandas]
```

Current status of implementation
--------------------------------
The SQL dialect is sqlite, with the addition of lists and sets.
 * To explode a list, use an ARRAY JOIN
 * manages WITH clause, subqueries, and of course group by, select, where, order by
 * What's known to miss: EXISTS, HAVING, Window functions, VALUES, many scalar and grouping function and probably more. Some of them are easy to implement
 * expressions containing agregation functions
 * dotted names (table.column) are limited to the selection of the table in a join, when there are several columns with the same name. Otherwise, remove the table prefix (that's due to the implementation of pandas.merge)
 * it's not possible to use a column of a table inside a subquery

License
-------
This work is licensed under the GPL v3.
