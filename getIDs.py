from __future__ import division
import numpy as np
import pandas as pd

class GetIDs():
    def __init__(self):
        df = pd.read_csv('/Users/Daniel/Documents/dataman/useraccess/processed/Daniel_Taylor_356/standard_data.csv')

        self.IDs = []
        coil = np.zeros(len(df['Coil']))
        self.sex = np.zeros(len(df['Sex']))

        for cx, c in enumerate(df['Coil']):
            if not pd.isnull(c):
                coil[cx] = 1
            else:
                coil[cx] = c

        for sx, s in enumerate(df['Sex']):
            if not np.isnan(coil[sx]):
                if s == 'MALE':
                    self.sex[sx] = 1

        for ix, i in enumerate(df['CCID']):
            if not np.isnan(coil[ix]): self.IDs.append(i)

        self.ages = np.zeros(len(self.IDs))
        self.count = 0

        for ax, a in enumerate(df['Age']):
            if not np.isnan(coil[ax]):
                self.ages[self.count] = a
                self.count += 1

        x = np.argmax(self.ages)
        self.ages = np.delete(self.ages, x)
        self.IDs = np.delete(self.IDs, x)
        self.count += -1

    def getAll(self):
        return self.IDs, self.ages, self.count

    def getSexDistr(self):
        return self.sex

    def getRandomised(self):
        newPlaces = np.arange(self.count)
        np.random.shuffle(newPlaces)

        newIDs = np.zeros_like(self.IDs)
        newAges = np.zeros_like(self.ages)

        for i in range(self.count):
            newIDs[i] = self.IDs[newPlaces[i]]
            newAges[i] = self.ages[newPlaces[i]]
        return newIDs, newAges, self.count