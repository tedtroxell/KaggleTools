import numpy as np

np.random.seed(1)

DIMS = 2


X = np.random.randn( 1000, DIMS )

X[:-600] += np.pi+np.random.random( (400,DIMS) )
X[:-300] += np.pi+np.random.random( (700,DIMS) )

y = np.random.randn( 1000 )
y[:-600] += np.pi+np.random.random( 400 )
y[:-300] += np.pi+np.random.random( 700 )

from src.mode_splitter import ModeSplitter

ms = ModeSplitter()

ms.fit( X,y )
print(
    ms.predict( X[:10] )
)