x1 = 8
x2 = 5
y  = 64
alpha = 0.005

# Initial weights 
w1 = 2.0
w2 = 7.5

print(
    f"{'Iter':<4} {'w1':<12} {'w2':<12} "
    f"{'a':<12} {'cost':<14} {'dc/da':<12}"
)

for i in range(1, 100):

    # ----- Forward propagation -----
    a = x1*w1 + x2*w2

    cost = 0.5 * (a - y)**2

    # ----- Backward propagation -----
    dc_da = (a - y)         
    da_dw1 = x1
    da_dw2 = x2

    dc_dw1 = dc_da * da_dw1
    dc_dw2 = dc_da * da_dw2

    # ----- Weight update -----
    w1 = w1 - alpha * dc_dw1
    w2 = w2 - alpha * dc_dw2

    print(
        f"{i:<4} {w1:<12.9f} {w2:<12.9f} "
        f"{a:<12.9f} {cost:<14.9f} {dc_da:<12.9f}"
    )

# -------- Final result --------
print("\nFinal Weights:")
print("w1 =", w1)
print("w2 =", w2)
print("Final Output =", x1*w1 + x2*w2)
