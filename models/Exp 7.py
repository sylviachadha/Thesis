from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

# Generate an array of 100 random sample from a normal dist with
# mean 0 and stdv 1
random_sample = norm.rvs(loc=0,scale=1,size=100)
random_sample

# Distribution fitting
# norm.fit(data) returns a list of two parameters
# (mean, parameters[0] and std, parameters[1]) via a MLE approach
# to data, which should be in array form.
parameters = norm.fit(random_sample)
print("Parameters of fitted distribution on sample using MLE are "),parameters

# 100 numbers b/w -5 and 5
x = np.linspace(-5,5,100)
x

# Generate the pdf (fitted distribution)
fitted_pdf = norm.pdf(x,loc = parameters[0],scale = parameters[1])
# Generate the pdf (normal distribution non fitted)
normal_pdf = norm.pdf(x)

# Type help(plot) for a ton of information on pyplot
plt.plot(x,fitted_pdf,"red",label="Fitted normal dist",linestyle="dashed", linewidth=2)
plt.plot(x,normal_pdf,"blue",label="Normal dist", linewidth=2)
plt.hist(random_sample,color="cyan",alpha=.3) #alpha, from 0 (transparent) to 1 (opaque)
plt.title("Normal distribution fitting")
# insert a legend in the plot (using label)
plt.legend()

plt.show()