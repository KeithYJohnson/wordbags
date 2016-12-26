from load_data import *
from params import KMEANS_DICT

def inspect_clusters(word_centroid_map):
    for cluster in range(0,10):
        #
        # Print the cluster number
        print("\nCluster %d" % cluster)
        #
        # Find all of the words for that cluster number, and print(them out)
        # words = []
        words = [k for k, w in word_centroid_map.items() if w == cluster]
        print(words)


word_centroid_map = load_data(KMEANS_DICT)
inspect_clusters(word_centroid_map)
# >>>
# Cluster 0
# ['pastor', 'formerly', 'fortune', 'banker', 'tycoon', 'patriarch', 'penniless', 'caleb']
#
# Cluster 1
# ['parade', 'breeze', 'bars', 'furniture', 'doors', 'horses', 'corners', 'trees', 'hallways', 'buildings', 'candles', 'rooms', 'trucks', 'windows', 'rocks', 'halls', 'statues', 'walls', 'overhead', 'smoke', 'floors', 'pipes', 'curtains', 'clouds', 'bushes', 'waves']
#
# Cluster 2
# ['generous', 'biased', 'unfair', 'picky', 'minority', 'harsh']
#
# Cluster 3
# ['burrows', 'excluding', 'isaac', 'schildkraut', 'scarlet', 'wolfe', 'robby', 'oates']
#
# Cluster 4
# ['passport', 'undress']
#
# Cluster 5
# ['ridicule', 'scorn']
#
# Cluster 6
# ['paranoia', 'warped', 'uncertainty', 'deception', 'tragedy', 'futility', 'decay']
#
# Cluster 7
# ['captivating', 'riveting', 'fascinating', 'dynamic']
#
# Cluster 8
# ['jaime', 'cummings', 'gallagher', 'dylan', 'napier', 'devine', 'richter', 'dyan', 'pepper', 'katt', 'glenda', 'steiger', 'moran', 'clarence', 'olin', 'bryan', 'swain', 'jenkins', 'stephens', 'garcia', 'byrne', 'mahoney', 'sutton', 'bergen', 'bobbie', 'weber', 'egan']
#
# Cluster 9
# ['rack', 'demand']
