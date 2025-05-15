from plavchan_gpu.plavchan_gpu import plavchan_periodogram
import numpy as np
import matplotlib.pyplot as plt
from time import time

np.random.seed(0)

times = [np.random.uniform(0, 30, 500)]
mags = [(np.sin(times[0]) + np.sin(times[0]*0.13) + np.random.normal(0, 0.1, 500))]
times = [list(t) for t in times]
mags = [list(m) for m in mags]

trialperiods = np.linspace(0.05, 10, 100000).tolist()
width = 0.05

t1 = time()
pgram = plavchan_periodogram(mags, times, trialperiods, width)[0]
t2 = time()
print("GPU time:", t2-t1)


# write out a plot of the periodogram results
plt.plot(times[0], mags[0], ".")
plt.title("Lightcurve")
plt.savefig("./out_Test_lc.png")
plt.clf()

# print a folded lightcurve on the best period
best = trialperiods[np.argmax(pgram)]
plt.plot(np.array(times[0]) % best, mags[0], ".")
plt.title("Folded Lightcurve")
plt.savefig("./out_Test_folded.png")
plt.clf()

plt.plot(trialperiods, pgram)
plt.savefig("./out_Test.png")
plt.clf()

sorted = np.argsort(pgram)[-10:][::-1]
print("Top 10 periods:")
for i in sorted:
    print(trialperiods[i], pgram[i])



