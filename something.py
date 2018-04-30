import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse
from scipy import sparse

customers = pd.read_csv('/home/rubione/work/public/data/customers.csv')
products = pd.read_csv('/home/rubione/work/public/data/products.csv')
transactions = pd.read_csv('/home/rubione/work/public/data/transactions.csv')

def gen_occurance_matrix(date):
    transactions["customer_hash"]=pd.factorize(transactions.customer_id)[0]
    transactions["product_hash"]=pd.factorize(transactions.product_id)[0]
    transactions.transaction_date=pd.to_datetime(transactions.transaction_date)
    customers2=transactions.customer_hash.values[transactions.transaction_date<parse(date)]
    products2=transactions.product_hash.values[transactions.transaction_date<parse(date)]
    number_of_transactions=np.shape(products2)[0]
    occurence_matrix=sparse.csr_matrix((np.ones(number_of_transactions),
                  (customers2,products2)),
                    shape=(np.max(customers2)+1,np.max(products2)+1))
    return occurence_matrix
def gen_correlsations(A):
    N=A.shape[1]
    C=((A.T*A -(sum(A).T*sum(A)/N))/(N-1)).todense()
    V=np.sqrt(np.mat(np.diag(C)).T*np.mat(np.diag(C)))
    COV = np.divide(C,V+1e-119)
    return COV

date='2010-08-01 00:00:00'
A=gen_occurance_matrix(date)
print("A is done")
COV=gen_correlsations(A)

np.save("covmatrix",COV)
cov_matr=np.load("covmatrix.npy")

transactions["customer_hash"]=pd.factorize(transactions.customer_id)[0]
transactions["product_hash"]=pd.factorize(transactions.product_id)[0]
transactions.transaction_date=pd.to_datetime(transactions.transaction_date)
old_subsample=transactions[transactions.transaction_date<parse('2010-08-01 00:00:00')]
new_subsample=transactions[transactions.transaction_date>parse('2010-08-01 00:00:00')]

COV2=np.fill_diagonal(COV,0)
for i in np.unique(transactions.product_hash.values):
    old_customers=old_subsample[old_subsample.product_hash==i].customer_hash.values
    new_period_customers=new_subsample[new_subsample.product_hash==i].customer_hash.values
    new_customers=list(set(new_period_customers)-set(old_customers))
    customer_slice=A[list(set(range(A.shape[0]))-set(old_customers)),:]
    product_slice=COV[:,i]
    product_slice[product_slice<0]=0
    product_slice=np.power(product_slice,0.3)*1
    scores=customer_slice.dot(product_slice)
    scores=np.array(scores)
    scores=np.reshape(scores,np.shape(scores)[0])
    ids=list(scores.argsort()[-len(new_customers)*5:][::-1])
    in_both=list(set(ids).intersection(set(new_customers)))
    try:
        accuracy=len(in_both)/len(new_customers)
    except:
        print("no new customers")
    print([i,accuracy,len(new_customers)])
    
def performance_metric(product_id,exponent,multiplicator):
    i=product_id
    old_customers=old_subsample[old_subsample.product_hash==i].customer_hash.values
    new_period_customers=new_subsample[new_subsample.product_hash==i].customer_hash.values
    new_customers=list(set(new_period_customers)-set(old_customers))
    customer_slice=A[list(set(range(A.shape[0]))-set(old_customers)),:]
    product_slice=COV[:,i]
    product_slice[product_slice<0]=0
    product_slice=np.power(product_slice,exponent)*multiplicator
    scores=customer_slice.dot(product_slice)
    scores=np.array(scores)
    scores=np.reshape(scores,np.shape(scores)[0])
    ids=list(scores.argsort()[-len(new_customers)*3:][::-1])
    in_both=list(set(ids).intersection(set(new_customers)))
    try:
        accuracy=len(in_both)/len(new_customers)
        return(accuracy)
    except:
        print("no new customers")
        return(-10000)
performance_metric(81,0.3,1)

exponents=np.linspace(0.05,2,20)
multiplicators=np.linspace(1,1,1)
for exponent in exponents:
    for multiplicator in multiplicators:
        print([exponent,multiplicator,performance_metric(81,exponent,multiplicator)])
