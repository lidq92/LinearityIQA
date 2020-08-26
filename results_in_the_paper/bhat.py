import re
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['pdf.fonttype'] = 42
 
file_object = open('bhat.log','rU', encoding='UTF-8')
bhat = []
try:
    for line in file_object:
        g = re.search("bhat.*", line)
        if g:
            # print(g.group())
            # f.writelines(line[7:])
            bhat.append(float(line[7:]))


finally:
	file_object.close()

print(len(bhat))

plt.plot(np.arange(1,26491), bhat[0::3])
plt.xlabel("iteration") 
plt.ylabel(r"$\hat{b}$")
plt.savefig('bhat.pdf',bbox_inches='tight') 
plt.close()

# plt.plot(np.arange(1,26491), bhat[1::3])
# plt.xlabel("iteration") 
# plt.ylabel(r"$\hat{b}$")
# plt.savefig('bhat4.pdf',bbox_inches='tight') 
# plt.close()

# plt.plot(np.arange(1,26491), bhat[2::3])
# plt.xlabel("iteration") 
# plt.ylabel(r"$\hat{b}$")
# plt.savefig('bhat5.pdf',bbox_inches='tight') 
# plt.close()

Iters = 883

avgbhat = []
tmp = 0
k = 1
for tmp1 in bhat[0::3]:
	tmp += tmp1
	k += 1
	if k>Iters:
		avgbhat.append(tmp/Iters)
		tmp = 0
		k = 1
print('tmp={},k={}.'.format(tmp,k))
plt.plot(np.arange(1,31), avgbhat)
plt.xlabel("epoch") 
plt.ylabel(r"average $\hat{b}$")
plt.savefig('avgbhat.pdf',bbox_inches='tight') 
plt.close()

# avgbhat = []
# tmp = 0
# k = 1
# for tmp1 in bhat[1::3]:
# 	tmp += tmp1
# 	k += 1
# 	if k>Iters:
# 		avgbhat.append(tmp/Iters)
# 		tmp = 0
# 		k = 1
# print('tmp={},k={}.'.format(tmp,k))
# plt.plot(np.arange(1,31), avgbhat)
# plt.xlabel("epoch") 
# plt.ylabel(r"average $\hat{b}$")
# plt.savefig('avgbhat4.pdf',bbox_inches='tight') 
# plt.close()


# avgbhat = []
# tmp = 0
# k = 1
# for tmp1 in bhat[2::3]:
# 	tmp += tmp1
# 	k += 1
# 	if k>Iters:
# 		avgbhat.append(tmp/Iters)
# 		tmp = 0
# 		k = 1
# print('tmp={},k={}.'.format(tmp,k))
# plt.plot(np.arange(1,31), avgbhat)
# plt.xlabel("epoch") 
# plt.ylabel(r"average $\hat{b}$")
# plt.savefig('avgbhat5.pdf',bbox_inches='tight') 
# plt.close()