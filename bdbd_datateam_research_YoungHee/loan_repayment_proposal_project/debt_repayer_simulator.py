from bokeh.models.widgets.inputs import TextAreaInput
import pandas as pd
import numpy as np
from scipy import stats
from collections import namedtuple
from tornado import gen
from functools import partial
from threading import Thread

from bokeh.layouts import layout, column, row
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, NumericInput, Slider, Div, SingleIntervalTicker, Button, TextAnnotation

from debt_repayer_MC import MCMDP_agent

doc = curdoc()

def make_beta_pdf(up, down, up_roll, down_roll):
    # up and down 으로 2개 a, b parameter 받아서 beta 함수 그리기
    up_beta = stats.beta(a=up[0], b=up[1])
    down_beta = stats.beta(a=down[0], b=down[1])
    x = np.linspace(-1, 1, 2000)
    up_pdf = np.roll(up_beta.pdf(x), up_roll)
    down_pdf = np.roll(down_beta.pdf(x), -down_roll)
    return x, up_pdf, down_pdf[::-1]

def update():
    dist_args = {
        'up': (up_a.value, up_b.value),
        'down': (down_a.value, down_b.value),
        'up_roll': up_roll.value,
        'down_roll': down_roll.value
    }
    
    x, up_pdf, down_pdf = make_beta_pdf(**dist_args)
    source.data['x'] = x
    source.data['up_pdf'] = up_pdf
    source.data['down_pdf'] = down_pdf

def calc():
    start_debt = 5000
    interest_rate = 0.0375
    dist_args = {
        'up_a': up_a.value, 
        'up_b': up_b.value,
        'down_a': down_a.value, 
        'down_b': down_b.value,
        'up_roll': up_roll.value / 2000,
        'down_roll': down_roll.value / 2000,
    }

    agent = MCMDP_agent(start_debt, interest_rate, dist_args)
    eps = 1
    for i in range(30000):
        eps = max(eps * 0.9997, 0.0001)
        _policy = agent.policy_improve(eps)
        cum_reward = agent.policy_evaluate(_policy)
        doc.add_next_tick_callback(partial(add_log, x=i, y=cum_reward))

@gen.coroutine
def add_log(i, g):
    log_source.stream(dict(x=[i], y=[g]))
    

# Set widgets
title = Div(text='<h1 style="text-align: left">코인으로 마통갚기 강화학습 시뮬레이터</h1>', height=50)
desc_problem = Div(text="""<b>주어진 문제 : </b>30대 직장인 남성 A씨는 최근 코인으로 돈을 벌었다는 친구들의 이야기를 듣고 코인 투자를 하려고 한다.
                            A씨는 다달이 월급에서 생활비를 제외한 <i>60<i>만원 중에 일부를 코인에 투자하려고 한다. 
                            A씨는 동시에 코인 투자를 통해서 마이너스 통장에 <i>5000<i>만원의 대출을 갚고 싶다.
                            매달 <i>60<i>만원 중 얼마를 투자하고 얼마를 대출을 갚으면 좋을까?""",
                    height=40)
                         
up_a = Slider(title="상승 alpha", value=1.5, start=1.25, end=10, step=0.25, )
up_b = Slider(title="상승 beta", value=6, start=1.25, end=10, step=0.25 )
down_a = Slider(title="하락 alpha", value=1.5, start=1.25, end=10, step=0.25 )
down_b = Slider(title="하락 beta", value=6, start=1.25, end=10, step=0.25 )
up_roll = Slider(title="상승 수평 롤링", value=20, start=-200, end=200, step=10 )
down_roll = Slider(title="하락 수평 롤링", value=20, start=-200, end=200, step=10 )

debt_size = NumericInput(title="마이너스 통장 대출 금액", value=1000, low=0, high=10000)
interest_rate = NumericInput(title="대출이자", value=3.75, low=0.0, high=10.0, mode='float')
calc_button = Button(label="모델 학습하기", button_type="success")

progress = TextAreaInput(cols=2, rows=100, height=200)

controls = [up_a, up_b, down_a, down_b, up_roll, down_roll]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

calc_button.on_click(calc)

# set data and plot
source = ColumnDataSource()
log_source = ColumnDataSource()

p = figure(x_range=[-1, 1], title="코인 투자 변동성에 대한 가상분포 설정", height=320, width=600, background_fill_color="#fafafa")
p.line(x='x', y='up_pdf', source=source, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="상승 확률분포")
p.line(x='x', y='down_pdf', source=source, line_color="orange", line_width=4, alpha=0.7, legend_label="하락 확률분포")
p.y_range.start = 0
p.legend.location = "center_right"
p.legend.background_fill_color = "#fefefe"
p.xaxis.ticker = SingleIntervalTicker(interval=0.1)
p.xaxis.axis_label = '코인 상승률 (%)'
p.yaxis.axis_label = 'Pr(x)'
p.grid.grid_line_color="white"


dist_widgets = column(up_a, up_b, down_a, down_b, up_roll, down_roll, sizing_mode='fixed')
agent_widgets = column(debt_size, interest_rate, calc_button, sizing_mode='fixed')

l = layout(title, 
           desc_problem,
           [dist_widgets, p], 
           [agent_widgets, progress], 
           sizing_mode='stretch_width')

update()

doc.add_root(l)
doc.title = "코인으로 마통갚기 시뮬레이션"