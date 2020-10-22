import numpy as np
import scipy.sparse

class DatasetGen():
    def __init__(self, nb_item, nb_user):
        self.nb_item = nb_item
        self.nb_user = nb_user
    
    def create_item_group(self, nb_item, grp_size, overlap=None):
        self.grp_size = grp_size
        if isinstance(grp_size, int):
            self.grp_size = [self.nb_item//grp_size for _ in range(grp_size)]
            for i in range(self.nb_item%grp_size):
                self.grp_size[i] += 1
        else:
            if sum(grp_size) < self.nb_item:
                raise ValueError('Certains items n\'appartiennent à aucun groupe !')
        self.item_grp = []
        item = np.arange(self.nb_item)
        np.random.shuffle(item)
        i = 0
        for k in np.cumsum(self.grp_size):
            self.item_grp.append(item[i:k])
            i=k

    def gen_rating_for_user(self, u, nb_rating):
        grp = self.user_grp[u]
        nb_rating = min(self.nb_item, nb_rating)
        rated_item = np.random.choice(self.nb_item, size=nb_rating, replace=False)
        rate = np.full_like(rated_item, 1)
        rate[np.isin(rated_item, self.item_grp[grp])] = 5
        return rated_item, rate

    def gen_dataset(self, grp_size, nb_rating, overlap=None):
        matrix = scipy.sparse.csr_matrix((self.nb_item, self.nb_user),  dtype=np.uint8)
        if nb_rating < self.nb_user:
            raise ValueError('Pas assez de note pour tous les utilisateurs')
        self.nb_rating = nb_rating

        self.create_item_group(self.nb_item, grp_size, overlap)

        self.user_grp = np.random.choice(len(self.grp_size), size=self.nb_user)

        avg_nb_rating = nb_rating//self.nb_user
        for u in range(self.nb_user):
            if nb_rating> self.nb_user-u:
                r = np.random.randint(1,2*avg_nb_rating+1)
            else:
                r = 1
            nb_rating -= r
            item, rating = self.gen_rating_for_user(u, r)
            matrix[item,u] = rating
        return matrix

