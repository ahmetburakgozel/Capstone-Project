from isyatirimhisse import StockData, Financials

# Uraz Akgül & Ahmet Burak Gözel

stock_data = StockData()
financials = Financials()

symbols = ['THYAO', 'PGSUS']  # Tek bir hisse senedi için tırnak içinde de verilebilir
start_date = '01-12-2017'  # Başlangıç tarihi belirtilmeli
end_date = '06-10-2023'  # Bitiş tarihi belirtilmezse sistem tarihini alır
exchange = '2'  # Hem TL hem de USD
frequency = '1mo'  # Aylık
observation = 'last'  # Son gözlem
return_type = '1'  # Logaritmik getiri
save_to_excel = True  # Excel dosyasına kaydet

df = stock_data.get_data(
    symbols=symbols,
    start_date=start_date,
    end_date=end_date,
    exchange=exchange,
    frequency=frequency,
    observation=observation,
    return_type=return_type,
    save_to_excel=save_to_excel
)

# pandas dataframe to excel
df.to_excel('isyatirimhissepackage/data.xlsx')
