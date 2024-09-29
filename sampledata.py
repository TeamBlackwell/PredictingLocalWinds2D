import numpy as np

# sample_data = np.array([((0,0), (30,10)), ((50,40), (10,40)), ((0,40),(2,12)),((50,40), (10,40)),((50,0), (10,4))])
sample_data = np.random.randint(0, 100, size=(100, 100, 2))
print(sample_data)
np.save('./data/wind_fields/3_m.npy', sample_data)