import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
#plt.rcParams['animation.ffmpeg_path'] = 'D:/portableSoftware/ShareX/ShareX/Tools/ffmpeg.exe'

interval = 50 # ms, time between animation frames

fig, ax = plt.subplots(figsize=(6,6))
plt.subplots_adjust(left=0.15, bottom=0.35)

ax.set_aspect('equal')
plt.xlim(-1.4*40,1.4*40)
plt.ylim(-1.4*40,1.4*40)
#plt.grid()
t = np.linspace(0, 2*np.pi, 400)
delta = 1
e =2
n=10
RD=40
rd=5
#plt.grid()
## pin
l = [ax.plot([], [], 'k-', lw=2)[0] for _ in range(41)]
## inner_pin
p = [ax.plot([], [], 'g-', lw=2)[0] for _ in range(10)]

def pin_init(pin):
    for i in range(len(pin)):
        pin[i].set_data([0], [0])

def pin_update(n,d,D):
    for i in range(int(n)):    
        x = (d/2*np.sin(t)+ D/2*np.cos(2*i*np.pi/n))
        y = (d/2*np.cos(t) + D/2*np.sin(2*i*np.pi/n))
        l[i].set_data(x, y)

def pin_update3(n,e,d,D,phi):
    for i in range(int(n)):    
        x = (d/2*np.sin(t)+ D/2*np.cos(2*i*np.pi/n))*np.cos(-phi/(n)) - (d/2*np.cos(t) + D/2*np.sin(2*i*np.pi/n))*np.sin(-phi/(n)) + e*np.cos(phi) 
        y = (d/2*np.sin(t)+ D/2*np.cos(2*i*np.pi/n))*np.sin(-phi/(n)) + (d/2*np.cos(t) + D/2*np.sin(2*i*np.pi/n))*np.cos(-phi/(n)) + e*np.sin(phi) 
        l[i].set_data(x, y)


def inner_pin_update(n,N,rd,Rd,phi):
    for i in range(int(n)):    
        x = (rd*np.sin(t)+ Rd*np.cos(2*i*np.pi/n))*np.cos(-phi/(N-1)) - (rd*np.cos(t) + Rd*np.sin(2*i*np.pi/n))*np.sin(-phi/(N-1))
        y = (rd*np.sin(t)+ Rd*np.cos(2*i*np.pi/n))*np.sin(-phi/(N-1)) + (rd*np.cos(t) + Rd*np.sin(2*i*np.pi/n))*np.cos(-phi/(N-1))
        p[i].set_data(x, y)

# draw 6 inner pins to start
for i in range(int(6)):
    x = (5*np.sin(t)+ 20*np.cos(2*i*np.pi/6))
    y = (5*np.cos(t) + 20*np.sin(2*i*np.pi/6))
    p[i] = ax.plot(x, y, 'g-')[0]

# Draw 10 pins to start
pin_update(10, 10, 80)

## draw drive_pin
a = 5*np.sin(t)
b = 5*np.cos(t) 
d0, = ax.plot(a, b,'k-', lw=2)

def drive_pin_update(r):
    x = r*np.sin(t)
    y = r*np.cos(t)
    d0.set_data(x,y)


#inner circle:
inner_circle = [ax.plot([], [], 'r-', lw=2)[0] for _ in range(10)]

for i in range(6):
    x = (rd+e)*np.cos(t)+0.5*RD*np.cos(2*i*np.pi/6)+e
    y = (rd+e)*np.sin(t)+0.5*RD*np.sin(2*i*np.pi/6)
    inner_circle[i] = ax.plot(x, y, 'r-')[0]

def update_inner_circle(e,n,N,rd,Rd, phi):
    
    for i in range(int(n)):
        x = ((rd+e)*np.cos(t)+Rd*np.cos(2*i*np.pi/n))*np.cos(-phi/(N-1)) - ((rd+e)*np.sin(t)+Rd*np.sin(2*i*np.pi/n))*np.sin(-phi/(N-1)) + e*np.cos(phi)
        y = ((rd+e)*np.cos(t)+Rd*np.cos(2*i*np.pi/n))*np.sin(-phi/(N-1)) + ((rd+e)*np.sin(t)+Rd*np.sin(2*i*np.pi/n))*np.cos(-phi/(N-1)) + e*np.sin(phi)
        inner_circle[i].set_data(x, y)
 
##inner pinA:
x = (rd+e)*np.cos(t)+e
y = (rd+e)*np.sin(t)
inner_pinA, = ax.plot(x,y,'r-')
##driver line and dot:
#self.line, = self.ax.plot([self.rd+self.e + self.e, 0],[0,0],'r-')
dotA, = ax.plot([-rd- e- e],[0], 'ro', ms=5)

def update_inner_pinA(e,Rm, phi):
    x = (Rm+e+e)*np.cos(t)+2*e*np.cos(phi)
    y = (Rm+e+e)*np.sin(t)+2*e*np.sin(phi)
    inner_pinA.set_data(x,y)
    
    x1 = (Rm+e+e)*np.cos(phi)+2*e*np.cos(phi)
    y1 = (Rm+e+e)*np.sin(phi)+2*e*np.sin(phi)
    #self.line.set_data([0,x1],[0,y1])
    dotA.set_data(x1, y1)

def ehypocycloid_common(e, n, D, d):
    # major radius
    RD=D/2
    # post radius
    rm=RD/n
    rd=d/2
    xa = RD*np.cos(t)-e*np.cos(n*t)
    ya = RD*np.sin(t)-e*np.sin(n*t)

    dxa = RD*(-np.sin(t)+(e/rm)*np.sin(n*t))
    dya = RD*(np.cos(t)-(e/rm)*np.cos(n*t))

    return (xa, ya, dxa, dya, rd)


def gen_ehypocycloidA(e,n,D,d, phis):
    xa, ya, dxa, dya, rd = ehypocycloid_common(e, n, D, d)

    x = (xa + rd/np.sqrt(dxa**2 + dya**2)*(-dya))*np.cos(-2*phis/(n-1))-(ya + rd/np.sqrt(dxa**2 + dya**2)*dxa)*np.sin(-2*phis/(n-1))  + 2*e*np.cos(phis) 
    y = (xa + rd/np.sqrt(dxa**2 + dya**2)*(-dya))*np.sin(-2*phis/(n-1))+(ya + rd/np.sqrt(dxa**2 + dya**2)*dxa)*np.cos(-2*phis/(n-1))  + 2*e*np.sin(phis)

    return x, y

def gen_ehypocycloidE(e,n,D,d, phis):
    """
    Outermost moving ring

    :param e: eccentricity of center thingie
    :paran n: number of posts
    :param D: Major diameter
    :param d: diameter of posts
    :param phis: phase
    """
    
    xa, ya, dxa, dya, rd = ehypocycloid_common(e, n, D, d)

    x = (xa - rd/np.sqrt(dxa**2 + dya**2)*(-dya))*np.cos(-2*phis/(n-1))-(ya - rd/np.sqrt(dxa**2 + dya**2)*dxa)*np.sin(-2*phis/(n-1))  + 2*e*np.cos(phis) 
    y = (xa - rd/np.sqrt(dxa**2 + dya**2)*(-dya))*np.sin(-2*phis/(n-1))+(ya - rd/np.sqrt(dxa**2 + dya**2)*dxa)*np.cos(-2*phis/(n-1))  + 2*e*np.sin(phis)

    return x, y

def gen_ehypocycloidD(e,n,D,d, phis):
    xa, ya, dxa, dya, rd = ehypocycloid_common(e, n, D, d)

    x = (xa - rd/np.sqrt(dxa**2 + dya**2)*(-dya))
    y = (ya - rd/np.sqrt(dxa**2 + dya**2)*dxa)

    return x, y

def gen_ehypocycloidF(e,n,D,d, phis):
    xa, ya, dxa, dya, rd = ehypocycloid_common(e, n, D, d)

    x = (xa + rd/np.sqrt(dxa**2 + dya**2)*(-dya))
    y = (ya + rd/np.sqrt(dxa**2 + dya**2)*dxa)

    return x, y

def init_ehypocycloids(e, n, D, d):
    global ehypocycloidA, ehypocycloidE, ehypocycloidD, ehypocycloidF
    global edotA, edotD

    x, y = gen_ehypocycloidA(e, n, D, d, 0)
    ehypocycloidA = ax.plot(x,y,'r-')[0]
    edotA = ax.plot([RD - rd],[0], 'ro', ms=5)[0]

    x, y = gen_ehypocycloidE(e, n, D, d, 0)
    ehypocycloidE = ax.plot(x,y,'r-')[0]

    x, y = gen_ehypocycloidD(e, n, D, d, 0)
    ehypocycloidD = ax.plot(x,y,'b-')[0]
    edotD = ax.plot([RD - rd + e],[0], 'bo', ms=5)[0]

    x, y = gen_ehypocycloidF(e, n, D, d, 0)
    ehypocycloidF = ax.plot(x, y, 'b-')[0]


axcolor = 'lightgoldenrodyellow'

ax_fm = plt.axes([0.25, 0.27, 0.5, 0.02], facecolor=axcolor)
ax_Rm = plt.axes([0.25, 0.24, 0.5, 0.02], facecolor=axcolor)
ax_n = plt.axes([0.25, 0.21, 0.5, 0.02], facecolor=axcolor)
ax_Rd = plt.axes([0.25, 0.18, 0.5, 0.02], facecolor=axcolor)
ax_rd = plt.axes([0.25, 0.15, 0.5, 0.02], facecolor=axcolor)
ax_e = plt.axes([0.25, 0.12, 0.5, 0.02], facecolor=axcolor)
ax_N = plt.axes([0.25, 0.09, 0.5, 0.02], facecolor=axcolor)
ax_d = plt.axes([0.25, 0.06, 0.5, 0.02], facecolor=axcolor)
ax_D = plt.axes([0.25, 0.03, 0.5, 0.02], facecolor=axcolor)

sli_fm = Slider(ax_fm, 'fm', 10, 100, valinit=50, valstep=delta)
sli_Rm = Slider(ax_Rm, 'Rm', 1, 10, valinit=5, valstep=delta)
sli_n = Slider(ax_n, 'n', 3, 10, valinit=6, valstep=delta)
sli_Rd = Slider(ax_Rd, 'Rd', 1, 40, valinit=20, valstep=delta)
sli_rd = Slider(ax_rd, 'rd', 1, 10, valinit=5, valstep=delta)
sli_e = Slider(ax_e, 'e', 0.1, 10, valinit=1.4, valstep=delta/10)
sli_N = Slider(ax_N, 'N', 3, 40, valinit=10, valstep=delta)
sli_d = Slider(ax_d, 'd', 2, 20, valinit=10,valstep=delta)
sli_D = Slider(ax_D, 'D', 5, 100, valinit=80,valstep=delta)

def update(val):
    sfm = sli_Rm.val
    sRm = sli_Rm.val
    sRd = sli_Rd.val
    sn = sli_n.val
    srd = sli_rd.val    
    se = sli_e.val
    sN = sli_N.val
    sd = sli_d.val
    sD = sli_D.val
    ax.set_xlim(-1.4*0.5*sD,1.4*0.5*sD)
    ax.set_ylim(-1.4*0.5*sD,1.4*0.5*sD)



sli_fm.on_changed(update)
sli_Rm.on_changed(update)
sli_Rd.on_changed(update)
sli_n.on_changed(update)
sli_rd.on_changed(update)
sli_e.on_changed(update)
sli_N.on_changed(update)
sli_d.on_changed(update)
sli_D.on_changed(update)

resetax = plt.axes([0.85, 0.01, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    sli_fm.reset()    
    sli_Rm.reset()
    sli_n.reset()
    sli_rd.reset()
    sli_Rd.reset()    
    sli_e.reset()
    sli_N.reset()
    sli_d.reset()
    sli_D.reset()

button.on_clicked(reset)

init_ehypocycloids(2, 10, 80, 10)

def animate(frame):
    sfm = sli_fm.val
    sRm = sli_Rm.val
    sRd = sli_Rd.val
    sn = sli_n.val
    srd = sli_rd.val    
    se = sli_e.val
    sN = sli_N.val
    sd = sli_d.val
    sD = sli_D.val
    frame = frame+1
    phi = 2*np.pi*frame/sfm


    # init pins
    pin_init(l)
    # init outer pins
    pin_init(p)
    pin_init(inner_circle)
    pin_update3(sN,se,sd,sD,phi)
    update_inner_pinA(se,sRm, phi)
    #update_inner_pinD(se,sRm, phi)
    #inner_pin_update(sn,sN,srd,sRd,phi)
    #drive_pin_update(sRm)
    #update_inner_circle(se,sn,sN,srd,sRd, phi)
    
    x, y = gen_ehypocycloidA(se,sN,sD,sd, phi)
    ehypocycloidA.set_data(x, y)
    edotA.set_data(x[0], y[0])

    x, y = gen_ehypocycloidE(se,sN,sD,sd, phi)
    ehypocycloidE.set_data(x, y)

    x, y = gen_ehypocycloidD(se,sN,sD,sd, phi)
    ehypocycloidD.set_data(x, y)
    edotD.set_data(x[0], y[0])

    x, y = gen_ehypocycloidF(se,sN,sD,sd, phi)
    ehypocycloidF.set_data(x, y)

    fig.canvas.draw_idle()



ani = animation.FuncAnimation(fig, animate,frames=sli_fm.val*(sli_N.val-1), interval=interval)
dpi=100
##un-comment the next line, if you want to save the animation as gif:
#hypo.animation.save('myhypocycloid.gif', writer='pillow', fps=10, dpi=75)
#ani.save('myGUI1.mp4', writer="ffmpeg",dpi=dpi)
plt.show()