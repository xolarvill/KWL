import pandas as pd

def houseprice(raw_excel_loc, geodata_loc):
    """
    Processes house price data from a raw Excel file and updates a geodata Excel file with the average house prices.
    Args:
        raw_excel_loc (str): The file path to the raw Excel file containing house price data.
        geodata_loc (str): The file path to the geodata Excel file to be updated with house prices.
    The raw Excel file should contain house price data with columns for each year (e.g., '2010年', '2012年', etc.) and a column for provinces ('省份').
    The geodata Excel file should contain columns 'provname' for province names and 'year' for the year.
    The function calculates the average house price for each province and year, and updates the geodata file with these values in the 'houseprice' column.
    """
    raw_excel = pd.read_excel(raw_excel_loc)
    
    values = []
    year = ['2010年','2012年','2014年','2016年','2018年','2020年','2022年']
    for t in year:
        province_group = raw_excel.groupby('省份')
        avg_value = province_group[t].mean()
        for province, value in avg_value.items():
            values.append((province, t, value))

    geodata = pd.read_excel(geodata_loc)
    for province, t, value in values:
        year_num = int(t[:-1])  # Extract the year number from the string
        geodata.loc[(geodata['provname'] == province) & (geodata['year'] == year_num), 'houseprice'] = value

    geodata.to_excel(geodata_loc, index=False)


