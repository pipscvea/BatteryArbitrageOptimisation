from joblib import load
# from training import X_test
import strategy
import matplotlib.pyplot as plt

loaded_model = load("refined_model1_2017_JantoMarch.joblib")
strategy.df['Predicted_Action'] = loaded_model.predict(strategy.X)

profit = 0

for i, row in strategy.df.iterrows():
    action = row['Predicted_Action']
    price = row['SystemSellPrice'] * 0.001  # Convert to £/kWh
    
    if action == 'Strong Buy' and strategy.charge < (strategy.battery_capacity - strategy.StrongBuyVol):
        strategy.charge += strategy.StrongBuyVol * strategy.efficiency
        profit -= price * strategy.StrongBuyVol
        print(f"Strong Buy: {strategy.charge:.2f} kWh, Profit: {profit:.2f} £")

    elif action == 'Mid Buy' and strategy.charge < (strategy.battery_capacity - strategy.MidBuyVol):
        strategy.charge += strategy.MidBuyVol * strategy.efficiency
        profit -= price * strategy.MidBuyVol
        print(f"Mid Buy: {strategy.charge:.2f} kWh, Profit: {profit:.2f} £")
    
    elif action == 'Weak Buy' and strategy.charge < (strategy.battery_capacity - strategy.WeakBuyVol):
        strategy.charge += strategy.WeakBuyVol * strategy.efficiency
        profit -= price * strategy.WeakBuyVol
        print(f"Weak Buy: {strategy.charge:.2f} kWh, Profit: {profit:.2f} £")

    elif action == 'Weak Sell' and (strategy.charge + strategy.WeakSellVol) > 0:
        strategy.charge -= strategy.WeakSellVol
        profit += price * strategy.WeakSellVol
        print(f"Weak Sell: {strategy.charge:.2f} kWh, Profit: {profit:.2f} £")

    elif action == 'Mid Sell' and (strategy.charge + strategy.MidSellVol) > 0:
        strategy.charge -= strategy.MidSellVol
        profit += price * strategy.MidSellVol
        print(f"Mid Sell: {strategy.charge:.2f} kWh, Profit: {profit:.2f} £")

    elif action == 'Strong Sell' and (strategy.charge + strategy.StrongSellVol) > 0:
        strategy.charge -= strategy.StrongSellVol
        profit += price * strategy.StrongSellVol
        print(f"Strong Sell: {strategy.charge:.2f} kWh, Profit: {profit:.2f} £")

    elif action == 'HOLD':
        strategy.charge += 0
        profit += 0
        print(f"HOLD: {strategy.charge:.2f} kWh, Profit: {profit:.2f} £")
    strategy.charge_list.append(strategy.charge)
    strategy.profit_list.append(profit)


print(f"Simulated Profit: £{profit:.2f}")
print(f"Battery Charge: {strategy.charge:.2f} kWh")

fig, ax = plt.subplots(1,3,figsize=(12, 6))
ax[0].plot(strategy.df.index, strategy.charge_list, label='Battery Charge (kWh)', color='blue')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Charge (kWh)')
ax[0].set_title('Battery Charge Over Time')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(strategy.df.index, strategy.profit_list, label='Profit (£)', color='green')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Profit (£)')
ax[1].set_title('Profit Over Time')
ax[1].legend()
ax[1].grid(True)

ax[2].plot(strategy.df.index, strategy.df['SystemSellPrice'], label='SSP', color='orange')
ax[2].plot(strategy.df.index, strategy.df['Demand']*0.0012, label='Demand', color='red')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('SSP (£)')
ax[2].set_title('SSP Over Time')
ax[2].legend()
ax[2].grid(True)
fig.savefig("figs\\Charge_Profit_SSP&DemandSubPlot_refined_model1_2017_JantoMarch.jpg")



###### test trained model to see how it would have performed on data