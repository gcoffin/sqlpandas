import typeguard
import sqldf
import sqlparser
import pypama

for name, fun in list(sqldf.__dict__.items()):
    if isinstance(fun, type) or callable(fun) and not isinstance(fun, pypama.Pattern):
        fun = typeguard.typechecked(fun)
        setattr(sqldf, name, fun)


for name, fun in list(sqlparser.__dict__.items()):
    if name in ["TClass", "IClass", "KClass"]:
        continue
    if isinstance(fun, (sqlparser.TClass, sqlparser.KClass, sqlparser.IClass)):
        continue
    if isinstance(fun, type) or callable(fun):
        fun = typeguard.typechecked(fun)
