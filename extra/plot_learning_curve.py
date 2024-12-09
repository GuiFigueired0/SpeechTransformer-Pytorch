import os
import pandas as pd 
import matplotlib.pyplot as plt
        
def plot_learning_curve(batch_stats, save=False, show=False):
    plt.figure(figsize=(10, 5))
    plt.plot(batch_stats['accuracy'], label="Accuracy")
    plt.plot(batch_stats['loss'], label="Loss")
    plt.title("Learning Curve")
    plt.xlabel("Batch (every 100 steps)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig("learning_curve.png")
        print("Learning curve saved as 'learning_curve.png'.")
    if show:
        plt.show()
    
def main():
    learning_curve_path = os.path.join(os.getcwd(), f"learning_curve.csv")
    raw_csv_data = pd.read_csv(learning_curve_path)
    df = raw_csv_data.copy()
    plot_learning_curve(df, save=True, show=False)

if __name__ == "__main__":
    main()