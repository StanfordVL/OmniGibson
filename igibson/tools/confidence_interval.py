import numpy as np
import scipy.stats as st

#define sample data
data = [0.46, 0.46, 0.7, 0.58, ]

#create 95% confidence interval for population mean weight
output = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
print(output)