import bnlearn as bn
from pgmpy.estimators.CITests import chi_square,g_sq
from causallearn.utils.cit import CIT
bifile = 'data_bif/mildew.bif'
num_samples = 1000
model = bn.import_DAG(bifile,verbose=0)
adj_matrix = model['adjmat']
print(model['model'].nodes())
print(type(adj_matrix))
print(adj_matrix)
print(model['model'].edges())
bn.sampling()
data_sample = bn.sampling(model, n=num_samples,verbose=0)

# print(data_sample.head())
# print(data_sample.info())
# Compute edge strength with chi square test
# re_sum_pgmpy = 0
# re_sum_causallearn = 0
# alpha = 0.05
# num_samples = 10000
# for i in range(1,11):
#     data_sample = bn.sampling(model, n=num_samples,verbose=0)
#     re1 = chi_square('asia','smoke',['either'],data_sample,significance_level=alpha)   # g_sq
#     # re1 = g_sq('asia','smoke',['either'],data_sample,significance_level=alpha)
#     print(f'aisa prep smoke | either ind= {re1}')
#     re2 = chi_square('asia','smoke',[],data_sample,significance_level=alpha)
#     # re2 = g_sq('asia','smoke',[],data_sample,significance_level=alpha)
#     print(f'aisa prep smoke, ind= {re2}')
#     if re1 == False and re2 == True:
#         re_sum_pgmpy += 1

#     ci_test = CIT(data_sample.to_numpy(copy=True), method='chisq', alpha=alpha)   # gsq  chisq
#     X = 'asia'
#     Y = 'smoke'
#     X = data_sample.columns.get_loc(X)
#     Y = data_sample.columns.get_loc(Y)
#     # print(X,Y)

#     condition_set = [data_sample.columns.get_loc('either')]
#     re1 = ci_test(X, Y, condition_set)
#     print(f'asia prep smoke  ind= {re1}')

#     condition_set = []
#     re2 = ci_test(X, Y, condition_set)
#     print(f'asia prep smoke  ind= {re2}')
#     if re1 < alpha and re2 >= alpha:
#         re_sum_causallearn += 1

# print(f'pgmpy: {re_sum_pgmpy}')
# print(f'causallearn: {re_sum_causallearn}')

# asia_learned = bn.structure_learning.fit(data_sample, methodtype='pc')

# print(asia_learned['model'].edges())
# print(asia_learned['model'].nodes())

# bn.plot(asia_learned)