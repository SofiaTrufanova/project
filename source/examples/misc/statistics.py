import numpy as np
import scipy.stats as sps
from scipy.stats import combine_pvalues
import pingouin as pg

class MultivariateNormalTests:
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def __call__(self, X, test_name='shapiro-wilk', combining_method='fisher'):
        '''
        X is the numpy array with size (n, d), where n is the number of samples, d is the sample dimension
        '''
        if test_name == 'henze-zirkler':
            p_val = pg.multivariate_normality(X, alpha=self.alpha)[1]
        if test_name == 'shapiro-wilk':
            assert combining_method in ['fisher', 'pearson', 'mudholkar_george', 
                                        'tippett', 'stouffer'], 'Wrong combinining p_value method'
            #p_values = sps.shapiro(X, axis=0).pvalue
            p_values = sps.normaltest(X, axis=0).pvalue
            p_val = combine_pvalues(p_values, method=combining_method).pvalue

        return p_val#, not (p_val < self.alpha)  


class MultiariateUniformTests:
    def __init__(self, loc=0, scale=1, alpha=0.05):
        '''
        Class for comparison to the distribution U[loc, loc + scale]
        '''
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.alpha = alpha

    def __call__(self, X, test_name='', combining_method='fisher'):
        '''
        X is the numpy array with size (n, d), where n is the number of samples, d is the sample dimension
        '''

        if test_name == 'ks_test':
            assert combining_method in ['fisher', 'pearson', 'mudholkar_george', 
                                        'tippett', 'stouffer'], 'Wrong combinining p_value method'

            p_values = sps.kstest(X, sps.uniform(loc=self.loc, scale=self.scale).cdf, axis=0).pvalue
            p_val = combine_pvalues(p_values, method=combining_method).pvalue

        return p_val, not (p_val < self.alpha)


class BootstrappedMultivariateNormalTest:
    def __init__(self, n_bootstraps: int=1024, subsempling_size: int=64) -> None:
        self.n_bootstraps = n_bootstraps
        self.subsempling_size = subsempling_size

    def __call__(self, X) -> float:
        subsempling_size = min(X.shape[0], self.subsempling_size)

        pvalues = np.zeros(self.n_bootstraps)
        for test_index in range(self.n_bootstraps):
            subsempling = X[np.random.choice(X.shape[0], size=subsempling_size, replace=False)]
            pvalues[test_index] = pg.multivariate_normality(subsempling)[1]

        return pvalues.mean()


class BootstrappedRandomProjectionUnivariateNormalTest:
    def __init__(self, test: callable=lambda X: sps.shapiro(X, axis=0).pvalue,
                 n_bootstraps: int=1024, subsempling_size: int=64) -> None:
        
        self.test = test
        self.n_bootstraps = n_bootstraps
        self.subsempling_size = subsempling_size

    def __call__(self, X) -> float:
        subsempling_size = min(X.shape[0], self.subsempling_size)

        pvalues = np.zeros(self.n_bootstraps)
        for test_index in range(self.n_bootstraps):
            projector = np.random.normal(size=X.shape[1])
            projector /= 1.0e-6 + np.linalg.norm(projector)
            
            subsempling = X[np.random.choice(X.shape[0], size=subsempling_size, replace=False)]
            subsempling = subsempling @ projector
            
            pvalues[test_index] = self.test(subsempling)

        return pvalues.mean() 