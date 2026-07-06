from src.nhanes_loader import load_master_dataframe


def main():
    master_df = load_master_dataframe(data_dir="data", rename_columns=True)

    print("\nPreview:")
    print(master_df.head())

    print("\nColumns:")
    print(master_df.columns.tolist())

    print("\nShape:")
    print(master_df.shape)


if __name__ == "__main__":
    main()