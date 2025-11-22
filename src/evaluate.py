# src/evaluate.py
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv('outputs/results.csv')
    print(df)
    # simple plot comparatif
    df.plot(x='model', y=['test_acc','test_loss'], kind='bar', secondary_y='test_loss')
    plt.title('Comparaison des performances (test_acc et test_loss)')
    plt.tight_layout()
    plt.savefig('outputs/figures/comparison.png')
    print('Saved outputs/figures/comparison.png')