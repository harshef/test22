import Orange

import random
random.seed(0)

som = Orange.projection.som.SOMLearner(map_shape=(3, 3),
                initialize=Orange.projection.som.InitializeRandom)
map = som(Orange.data.Table("iris.tab"))

print "Node    Instances"
print "\n".join(["%s  %d" % (str(n.pos), len(n.instances)) for n in map])

i, j = 0, 0
print


for i in range(3):
    for j in range(3):
        print "Data instances in cell (%d, %d):" % (i, j)
        for e in map[i, j].instances:
            #print map[i,j]
            print e