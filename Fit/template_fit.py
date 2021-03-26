import numpy as np
import scipy.optimize
    

class LocalTemplateFit:
    def __init__(self, cov, templates):
        self.V = cov
        self.T = np.array(templates)        
        self._param_mask = np.identity(self.T.shape[0])
        self._fixed_params = np.empty(self.T.shape[0])
        self._fixed_params[:] = np.nan

        
        
    def __call__(self, params):
        u = self.U(params)
        #b = np.linalg.solve(self.V, self.X - u)
        b = scipy.linalg.solve(self.V, self.X - u, assume_a='sym')
        return np.dot(self.X - u, b)

    def U(self, params):
        return np.dot(self._to_user_coords(params), self.T)

    def fix_template(self, template_idx, val):
        if np.isnan(self._param_mask.sum(axis=1)[template_idx]):
            self._fixed_params[template_idx] = val
        else:        
            self._param_mask = np.delete(self._param_mask, template_idx,axis=1)
            self._param_mask[:][template_idx] = np.nan
            self._fixed_params[template_idx] = val

    def release_template(self, template_idx):
        if np.isnan(self._param_mask.sum(axis=1)[template_idx]):
            self._param_mask = np.insert(self._param_mask, template_idx, np.zeros(self.T.shape[0]),axis=1)
            self._param_mask[:][template_idx] = 0
            self._param_mask[template_idx][template_idx] = 1
            self._fixed_params[template_idx] = np.nan

    def release_templates(self):
        self._fixed_parameters = np.empty(self.T.shape[0])
        self._fixed_params[:] = np.nan
        self._param_mask = np.identity(self.T.shape[0])

    
    def chisq(self, params, data):
        self.X = data
        return self(params)

    def minimize(self, data, *args, **kwargs):
        self.X = data
        bounds = [(0, 10) for _ in range(self._param_mask.shape[1])]
        results = scipy.optimize.dual_annealing(self, bounds, *args, local_search_options={'method': 'Newton-CG', 'jac': self._jacobian}, **kwargs)
        results.x = self._to_user_coords(results.x)
        return results

    def MINOS(self, data, *args, **kwargs):
        best_results = self.minimize(data, *args, **kwargs)
        start = best_results.x
        fun0 = best_results.fun

        errors = []
        error_def = 1
        minos_results = {}
        for i, s in enumerate(start):
            def func_MINOS(x):
                self.release_templates()
                self.fix_template(i, x)
                return self.minimize(data).fun - fun0 - error_def
    
            a = s
            b = s + 0.1
            f_a = func_MINOS(a)
    
            # find b such f(b) has the opposite sign of f(a)
            while True:        
                f_b = func_MINOS(b)
                if f_a * f_b < 0: break
                b += 0.1        

            up, results_up = scipy.optimize.bisect(func_MINOS, a, b, full_output=True)
            minos_results['e%d_up' % i] = results_up
    
            a = s
            b = s - 0.1
            # find b such f(b) has the opposite sign of f(a)
            while True:
                f_b = func_MINOS(b)
                if f_a * f_b < 0: break
                b -= 0.1

            dw, results_dw = scipy.optimize.bisect(func_MINOS, a, b, full_output=True)
            minos_results['e%d_dw' % i] = results_up

            errors.append((up, dw))

        self.release_templates()        
        minos_results['dx'] = errors

        best_results.update(minos_results)
        return best_results
    

    def _jacobian(self, params):
        u = self.U(params)
        #d1 = np.linalg.solve(self.V, self.X - u)
        #d2 = np.linalg.solve(self.V, self.T.transpose())
        d1 = scipy.linalg.solve(self.V, self.X - u, assume_a='sym')
        d2 = scipy.linalg.solve(self.V, self.T.transpose(), assume_a='sym')
        return self._to_optimizer_coords(-1 * np.dot(self.T, d1) - np.dot(self.X - u, d2))

    def _to_user_coords(self, optimizer_params):
        return np.nan_to_num(np.dot(self._param_mask, optimizer_params)) + np.nan_to_num(self._fixed_params)
                             
    def _to_optimizer_coords(self, user_params):
        return user_params[~np.isnan(self._param_mask.sum(axis=1))]

    
class GlobalTemplateFit(LocalTemplateFit):
    def __init__(self, cov, templates):
        """ 
        arguments:
           cov       : (NxN)   np.array
           templates : (txpxqxn) np.array 
                       where p x q x n = N
                       t is number of components
                       t x p x q = number of fit parameters
        """
        self.V = cov
        self.T = np.array(templates)

        if np.prod(self.T.shape[1:]) != self.V.shape[0]:
            raise TypeError(f'Template shape ({self.T.shape}) not compatible with covariance matrix ({self.V.shape})')
        
        self._param_mask = np.identity(np.prod(self.T.shape[:-1]))
        self._fixed_params = np.empty(np.prod(self.T.shape[:-1]))
        self._fixed_params[:] = np.nan

    def U(self, params):
        # flatten phase space
        flat_T = self.T.reshape((np.prod(self.T.shape[:-1]), self.T.shape[-1]))
        return np.dot(np.diag(self._to_user_coords(params)), flat_T).reshape((self.T.shape[0], np.prod(self.T.shape[1:]))).sum(axis=0)
    #return GlobalTemplateFit._mult(self.T, self._to_user_coords(params).reshape(self.T.shape[:-1])).reshape((self.T.shape[0], np.prod(self.T.shape[1:]))).sum(axis=0)


    def _mult(t, a):
        return np.dot(np.diag(a.flatten()), t.reshape((np.prod(a.shape), t.shape[-1]))).reshape(t.shape)
