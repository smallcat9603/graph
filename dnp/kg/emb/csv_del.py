import argparse
import pandas as pd

def parse_args():
    """
    Parse the arguments.

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="Transformer")

    parser.add_argument('--infile', default='infile.csv',
                        help='Input file. Default is "infile.csv". ')

    parser.add_argument('--outfile', default='outfile.txt',
                        help='Output file. Default is "outfile.txt".')

    return parser.parse_args()

def process(infile, outfile):
    # 读取CSV文件，同时去掉第一行（header=0表示第一行作为列名，因此跳过第一行数据）
    df = pd.read_csv(infile, header=0)

    # 去掉第一列，第二列和第四列
    df.drop(df.columns[[0, 1, 3]], axis=1, inplace=True)
    
    # 将处理后的数据保存到TXT文件
    df.to_csv(outfile, sep=' ', index=False, header=False)

def main():
    args = parse_args()
    process(args.infile, args.outfile)
    print(f'Transform done.')

if __name__ == "__main__":
    main()
