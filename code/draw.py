# 修改 matplotlib 坐标轴

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import numpy as np


#创建画布
fig = plt.figure(figsize=(16, 8))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)  
#将绘图区对象添加到画布中
fig.add_axes(ax)

#通过set_visible方法设置绘图区所有坐标轴隐藏
ax.axis[:].set_visible(False)

#ax.new_floating_axis代表添加新的坐标轴
ax.axis["x"] = ax.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax.axis["x"].set_axisline_style("->", size = 2.0)
#添加y坐标轴，且加上箭头
ax.axis["y"] = ax.new_floating_axis(1,0)
ax.axis["y"].set_axisline_style("->", size = 2.0)
#设置x、y轴上刻度显示方向
ax.axis["x"].set_axis_direction("bottom")
ax.axis["y"].set_axis_direction("left")

ax.axis["x"].label.set_text('Time(secs)')
ax.axis["y"].label.set_text('System response')


x=np.arange(0,10,0.1)

y1=np.exp(-1*x/(-3))
y2=np.exp(-1*x/(-5))
y3=np.exp(-1*x/(-10))
y4=np.ones(100)
y5=np.exp(-1*x/(10))
y6=np.exp(-1*x/(5))
y7=np.exp(-1*x/(3))

plt.plot(x,y1,marker='.',markersize=3,linestyle='dotted')
plt.plot(x,y2,marker='.',markersize=3,linestyle='dotted')
plt.plot(x,y3,marker='.',markersize=3,linestyle='dotted')
plt.plot(x,y4)
plt.plot(x,y5,marker='.',markersize=3,linestyle='dotted')
plt.plot(x,y6,marker='.',markersize=3,linestyle='dotted')
plt.plot(x,y7,marker='.',markersize=3,linestyle='dotted')

plt.ylim(ymin=0,ymax=4)
plt.xlim(xmin=0,xmax=10)
plt.show()
print(x.shape)