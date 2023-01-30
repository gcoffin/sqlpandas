
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


License
-------
This work is licensed under the GPL v3.
