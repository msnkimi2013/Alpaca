import pandas as pd
from datetime import date, datetime

from bokeh.layouts import layout, column
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, RadioButtonGroup, DatePicker, NumericInput, Slider, NumeralTickFormatter, DataTable, TableColumn, NumberFormatter, Div

from taxcalc import TransferTaxCalculator



def set_input_var_list(buy_date, buy_price, period, increase_rate):
    # set sell dates for simulation
    buy_date = datetime.strptime(buy_date, "%Y-%m-%d")
    sell_dates = [datetime(buy_date.year + i, buy_date.month, buy_date.day) for i in range(period+1)]
    
    # set sell prices for simulation
    sell_prices = [buy_price * increase_rate ** (i+1) for i in range(period+1)]
    
    # set living years for simulation
    living_years = [i for i in range(period+1)]

    return buy_date, sell_dates, sell_prices, living_years


def calc_tax_results(buy_date, buy_price, living: bool, n_house, restrict, share_ptg, period, increase_rate):
    buy_date, sell_dates, sell_prices, living_years = set_input_var_list(buy_date, buy_price, period, increase_rate)
    results = []
    for i in range(period+1):
        if living:
            tax = TransferTaxCalculator(buy_price*1e4, buy_date, sell_prices[i]*1e4, sell_dates[i], living_years[i], n_house, restrict, share_ptg)
            tax.build_tax()
            results.append(tax.get_verbose_process())
        else:
            tax = TransferTaxCalculator(buy_price*1e4, buy_date, sell_prices[i]*1e4, sell_dates[i], 0, n_house, restrict, share_ptg)
            tax.build_tax()
            results.append(tax.get_verbose_process())
    return results, sell_prices, sell_dates

def update():
    living = True if living_select.active == 0 else False
    n_house = n_house_select.active + 1
    restrict = True if restrict_select.active == 1 else False

    results, sell_prices, sell_dates = calc_tax_results(buy_date.value, buy_price.value, living, n_house, restrict, share_ptg.value*100, 20, increase_rate.value + 1)
    
    # 결과물 data로 넘기기
    df = pd.DataFrame(results)
    source.data = df
    source.data['Year'] = [d.year for d in sell_dates]
    source.data['Price'] = sell_prices
    source.data['Holding_years'] = source.data['index'] + 1
    source.data['Transfer_tax'] = source.data['양도소득세'] / 10000

# set widgets
buy_price = NumericInput(title="취득가격 (만원)", value=35000, low=0, high=1e+10)
buy_date = DatePicker(title="취득일자", value=date.today())
restrict_select = RadioButtonGroup(labels=['비조정대상지역', '조정대상지역'], active=0)
n_house_select = RadioButtonGroup(labels=['1주택', '2주택', '3주택'], active=0)
living_select = RadioButtonGroup(labels=['거주', '비거주'], active=0)
share_ptg = Slider(title="지분비율", value=1, start=0, end=1, step=0.01, format="0%")
increase_rate = Slider(title="연간가격상승률", value=0.1, start=0, end=1, step=0.01, format="0%")
#calc_button = Button(label="계산하기")

controls = [buy_price, buy_date, share_ptg, increase_rate]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

buttons = [restrict_select, n_house_select, living_select]
for button in buttons:
    button.on_change('active', lambda attr, old, new: update())

# set data and plot
source = ColumnDataSource()

TOOLTIPS = [
    ("연도", "@Year"),
    ("양도가격", "@Price{0.}"),
    ("보유기간", "@Holding_years"),
    ("양도세", "@Transfer_tax{0.}"),
]

p = figure(y_range=[0, 70000e+4], tooltips=TOOLTIPS, sizing_mode="stretch_both", height=400, width=600)
#p.line(x='연도', y='예상가격', source=source)
#p.scatter(x='연도', y='예상가격', source=source, fill_color='white')

p.line(x='Year', y='양도소득세', source=source, width=2)
p.scatter(x='Year', y='양도소득세', source=source, fill_color='white', size=8)


p.yaxis[0].formatter = NumeralTickFormatter(format="0.a")
p.xaxis.axis_label = "매각연도"
p.xaxis.axis_label_standoff = 20
p.yaxis.axis_label = "양도소득세금액 (단위: 백만원)"
p.yaxis.axis_label_standoff = 20

# set datatable
# whole data

columns = [
        TableColumn(field="양도가액", title="양도가액", formatter=NumberFormatter()),
        TableColumn(field="필요경비", title="필요경비", formatter=NumberFormatter()),
        TableColumn(field="양도차익", title="양도차익", formatter=NumberFormatter()),
        TableColumn(field="장기보유특별공제", title="장기보유특별공제", formatter=NumberFormatter()),
        TableColumn(field="양도소득금액", title="양도소득금액", formatter=NumberFormatter()),
        TableColumn(field="과세표준", title="과세표준", formatter=NumberFormatter()),
        TableColumn(field="세율", title="세율", formatter=NumberFormatter(format="0.00")),
        TableColumn(field="누진공제액", title="누진공제액", formatter=NumberFormatter()),
        TableColumn(field="양도소득세", title="양도소득세", formatter=NumberFormatter()),
    ]

# hover one time process data

table = DataTable(source=source, columns=columns, sizing_mode="stretch_width")

# set layouts
title = Div(text='<h1 style="text-align: left">양도소득세 시뮬레이터</h1>')

widgets = column(buy_price,
                 buy_date,
                 n_house_select,
                 restrict_select,
                 living_select,
                 share_ptg, 
                 increase_rate,
                 sizing_mode='fixed')

l = layout([title], [widgets, p], [table], sizing_mode='stretch_width')

update()

curdoc().add_root(l)
curdoc().title = "양도소득세 시뮬레이션"