import numpy as np
import time
from pyfirmata import Arduino

# Initialize Arduino connection
board = Arduino("COM7")
Step_Motor1, Dir_Motor1 = 10, 11
Step_Motor2, Dir_Motor2 = 8, 9
Step_Motor3, Dir_Motor3 = 6, 7
Step_Motor4, Dir_Motor4 = 4, 5

def readtext():
    """Read position data from a pre-saved .npy file."""
    data = None
    while data is None:
        try:
            data = np.load("PythonClient/data2.npy")
        except:
            pass
    return data

def moveMotor(tick, ts):
    """Move stepper motors based on tick values."""
    ticks = max(tick)
    for i in range(ticks):
        if i < tick[0]:
            board.digital[Step_Motor1].write(1)
        if i < tick[1]:
            board.digital[Step_Motor2].write(1)
        if i < tick[2]:
            board.digital[Step_Motor3].write(1)
        if i < tick[3]:
            board.digital[Step_Motor4].write(1)
        if i < ticks:
            board.digital[Step_Motor1].write(0)
        if i < ticks:
            board.digital[Step_Motor2].write(0)
        if i < ticks:
            board.digital[Step_Motor3].write(0)
        if i < ticks:
            board.digital[Step_Motor4].write(0)
        time.sleep(ts)

def motorMain(v, ts):
    """Set motor directions and execute movement."""
    Utemp = []
    for i in range(len(v)):
        board.digital[11 - (2 * i)].write(0 if v[i] < 0 else 1)
        Utemp.append(abs(v[i]))
    moveMotor(Utemp, ts)

def uvalues(u):
    """Generate sinusoidal control values."""
    t1 = np.linspace(1, u-1, num=u//2, endpoint=False)
    x1 = u * (np.sin(np.pi * t1 / (2 * u)))
    x1 = np.floor(x1)
    x2 = x1[::-1]
    return list(x1) + list(x2)

def umaker(UU, uall):
    """Generate motor control sequence."""
    u_ult = []
    for i in uall:
        for j in range(len(UU)):
            u_ult.append(int(i) if UU[j] > 0 else int(-i))
    return u_ult

def u_faultmaker(UUU, uall, motor):
    """Generate faulty motor control sequence."""
    u_ultfault = []
    for i in uall:
        for j in range(len(UUU)):
            val = int(i) if UUU[j] > 0 else int(-i)
            u_ultfault.append(0 if j == motor - 1 else val)
    return u_ultfault

def main():
    """Control motors and collect motion data."""
    # User inputs (hardcoded for consistency)
    u = 10  # Control amplitude
    num_round = 5  # Rounds before fault
    num_round_fault = 5  # Rounds with fault
    ts = 0.015  # Time step
    motor = 5  # Faulty motor
    elapseperiod = 0.0000015  # Elapse period

    # Generate control sequences
    uall = uvalues(u)
    u1_2 = [-u, 0, u, 0]
    u2_0 = [0, -u, -u, 0]
    u0_4 = [u, 0, 0, u]
    u4_3 = [-u, 0, u, 0]
    u3_0 = [0, 0, -u, -u]
    u0_1 = [u, u, 0, 0]
    U = (umaker(u1_2, uall) + umaker(u2_0, uall) + umaker(u0_4, uall) +
         umaker(u4_3, uall) + umaker(u3_0, uall) + umaker(u0_1, uall))
    U_fault = (u_faultmaker(u1_2, uall, motor) + u_faultmaker(u2_0, uall, motor) +
               u_faultmaker(u0_4, uall, motor) + u_faultmaker(u4_3, uall, motor) +
               u_faultmaker(u3_0, uall, motor) + u_faultmaker(u0_1, uall, motor))
    U = np.tile(U, num_round)
    U_fault = np.tile(U_fault, num_round_fault)
    U = np.array(list(U) + list(U_fault))

    # Reach starting point
    startu = umaker([u, u, 0, 0], uall)
    endu = umaker([-u, -u, 0, 0], uall)
    for i in range(len(startu) // 4):
        inpt = startu[i*4:(i+1)*4]
        motorMain(inpt, ts)
    time.sleep(1)

    # Main loop
    posxall, poszall, posyall = [], [], []
    u1all, u2all = [], []
    el, starttime = 0, time.time()
    while True:
        elapsetime = time.time() - starttime
        if elapsetime > elapseperiod:
            if el > 0:
                pos = 1000. * readtext()
                posxall.append(round(pos[0], 3))
                poszall.append(round(pos[2], 3))
                posyall.append(round(pos[1], 3))
            V = U[el:el+4]
            motorMain(V, ts)
            u1all.append(U[el])
            u2all.append(U[el + 1])
            el += 4
            starttime = time.time()
            if el > (len(U) - 4):
                pos = 1000. * readtext()
                posxall.append(round(pos[0], 3))
                poszall.append(round(pos[2], 3))
                posyall.append(round(pos[1], 3))
                break

    # Return to start
    for i in range(len(endu) // 4):
        inpt = endu[i*4:(i+1)*4]
        motorMain(inpt, ts)

    # Save data
    np.save("xpos_Fault3_motor4.npy", np.asarray(posxall))
    np.save("zpos_Fault3_motor4.npy", np.asarray(poszall))
    np.save("ypos_Fault3_motor4.npy", np.asarray(posyall))
    np.save("u1all_Fault3_motor4.npy", np.asarray(u1all))
    np.save("u2all_Fault3_motor4.npy", np.asarray(u2all))

if __name__ == "__main__":
    main()