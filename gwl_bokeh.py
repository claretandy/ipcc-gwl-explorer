import os

import pandas as pd
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, RangeSlider, MultiChoice, MultiSelect, CategoricalColorMapper, HoverTool, \
    CustomJS, Spinner, Div, Scatter, Dropdown, BoxSelectTool, CDSView, IndexFilter
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.layouts import column, row
from bokeh.plotting import figure, show
from bokeh.palettes import d3
from bokeh.transform import jitter, factor_cmap, factor_mark
from bokeh.core import enums
import numpy as np

# First, load all the data into a global variable
ukcp18_ts = pd.read_csv('UKCP18_GWLs_Yearly_Temperature_2022_10_13-10:28:31_AM.csv')
ukcp18_ts.drop(columns='Unnamed: 0', inplace=True)
ukcp18_ts['Ensemble_Name'] = 'UKCP18'

cmip5_ts = pd.read_csv('CMIP5_GWLs_Yearly_Temperature_2022_10_20-12:13:15_PM.csv')
cmip5_ts.drop(columns='Unnamed: 0', inplace=True)
cmip5_ts['Ensemble_Name'] = 'CMIP5'

cmip6_ts = pd.read_csv('CMIP6_GWLs_Yearly_Temperature_2022_10_14-10:48:48_PM.csv')
cmip6_ts.drop(columns='Unnamed: 0', inplace=True)
cmip6_ts['Ensemble_Name'] = 'CMIP6'

alldata = pd.concat([ukcp18_ts, cmip5_ts, cmip6_ts], ignore_index=True)
alldata.reset_index(inplace=True)
alldata.drop(columns='index', inplace=True)
all_ens_names = sorted(pd.unique(alldata['Ensemble_Name']))
palette = np.array(d3['Category10'][1 + len(all_ens_names)])[[0, 1, 3]].tolist()
fc = factor_cmap('Ensemble_Name', palette=palette, factors=all_ens_names)

# Set some plot markers
MARKERS = [mark for mark in enums.MarkerType if 'circle' in mark]
MARKERS.extend([mark for mark in enums.MarkerType if 'square' in mark])
MARKERS.extend([mark for mark in enums.MarkerType if 'triangle' in mark])


def get_dataset(df, ens_name, start_yr, end_yr, year_tol=0, exp_ids='all'):
    '''
    Function to retrieve model+scenario specific mean temperature for a given time period from a dataframe, allowing for some tolerance of missing years
    Returns a dataframe of records containing mean temperature where the model dates meet the criteria
    '''

    min_yrs = end_yr - start_yr - year_tol
    # print(ens_name, exp_ids, start_yr, end_yr)

    if exp_ids == 'all':
        df_period = df.loc[(df['Year'] >= start_yr) &
                           (df['Year'] < end_yr) &
                           (df['Ensemble_Name'] == ens_name)]
    else:
        df_period = df.loc[(df['Year'] >= start_yr) &
                           (df['Year'] < end_yr) &
                           (df['Ensemble_Name'] == ens_name) &
                           (df['Experiment_ID'].isin(exp_ids))]

    out_df = df_period.groupby(['Institution_ID', 'Source_ID', 'Variant_Label', 'Experiment_ID']).agg(
        Tmean=('Annual_Mean_Temp', "mean"), YearCount=('Year', pd.Series.nunique))
    df_new_ss = out_df.loc[out_df['YearCount'] >= min_yrs]
    df_new_ss.reset_index(inplace=True)

    return df_new_ss


def make_plot(baseline_period, future_period):

    p1 = figure(width=1400, height=600, x_range=sorted(np.unique(source.data['Experiment_ID'])), x_axis_label="Experiment", y_axis_label="Global Warming Level (°C)")
    p1.title.text = f"Global warming levels for the period {future_period[0]} to {future_period[1]}, with respect to a baseline period of {baseline_period[0]} to {baseline_period[1]}"
    renderer = p1.scatter(x=jitter("Experiment_ID", width=0.4, range=p1.x_range), y="Change", size=8, marker='circle',
               fill_alpha=0.5, line_color=None, fill_color=fc, legend_field='Ensemble_Name', source=source)

    # renderer.selection_glyph.update(fill_alpha=1, line_color="black", fill_color=fc)
    # renderer.nonselection_glyph.update(fill_alpha=0.2, line_color=None, fill_color=fc)
    renderer.selection_glyph = Scatter(fill_alpha=1, line_color="black", fill_color=fc)
    renderer.nonselection_glyph = Scatter(fill_alpha=0.2, line_color=None, fill_color=fc)
    print(f"renderer.selection_glyph: {renderer.selection_glyph}")
    print(f"renderer.nonselection_glyph: {renderer.nonselection_glyph}")

    p1.legend.location = "top_left"
    p1.legend.title = 'Ensemble Name'

    hover = HoverTool(
        tooltips=[
            ("Model", "@Source_ID"),
            ("Variant", "@Variant_Label"),
            ("Experiment", "@Experiment_ID"),
            ("Warming", "@Change{0.0}"),
        ],
        formatters={
            'Model': 'printf',
            'Variant': 'printf',
            'Experiment': 'printf'
        },
    )
    boxsel = BoxSelectTool(
        dimensions='height'
    )
    p1.add_tools(hover)
    p1.add_tools(boxsel)

    return p1, source, renderer


def printTable():
    columns = [TableColumn(field=col, title=col.replace('_', ' ')) for col in plot_source.data.keys()]
    i = plot_source.selected.indices
    if not i:
        my_data_table = DataTable(source=plot_source, columns=columns, fit_columns=True, width=1400, height=500)
    else:
        view = CDSView(source=plot_source, filters=[IndexFilter(i)])
        my_data_table = DataTable(source=plot_source, view=view, columns=columns, fit_columns=True, width=1400, height=500)

    return my_data_table


# All the sliders need to be loaded before the load_data function (because their values are used in that)
baseline_slider = RangeSlider(start=1850, end=2020, value=(1900, 1950), step=5, title="Baseline period")
baseline_slider.js_on_change("value", CustomJS(code="""
    console.log('baseline_slider: value=' + this.value, this.toString())
"""))

bl_tol = Spinner(title="Tolerance", low=0, high=20, step=1, value=10, width=80)
bl_tol.js_on_change("value", CustomJS(code="""
    console.log('bl_tol: value=' + this.value, this.toString())
"""))

future_slider = RangeSlider(start=2050, end=2100, value=(2080, 2100), step=5, title="Future period")
future_slider.js_on_change("value", CustomJS(code="""
    console.log('future_slider: value=' + this.value, this.toString())
"""))

fu_tol = Spinner(title="Tolerance", low=0, high=20, step=1, value=0, width=80)
fu_tol.js_on_change("value", CustomJS(code="""
    console.log('fu_tol: value=' + this.value, this.toString())
"""))

# Create all the other widgets (which are dependent on alldata being loaded)
exp_ids = sorted(pd.unique(alldata['Experiment_ID']).tolist())
scenario_choice = MultiChoice(title="Experiment Selection", value=exp_ids, options=exp_ids, width=390)
scenario_choice.js_on_change("value", CustomJS(code="""
    console.log('scenario_choice: value=' + this.value, this.toString())
"""))

models = sorted(pd.unique(alldata['Source_ID']).tolist())
model_choice = MultiSelect(title="Show Selected Models as Points", value=models, options=models, width=200, size=6)
model_choice.js_on_change("value", CustomJS(code="""
    console.log('scenario_choice: value=' + this.value, this.toString())
"""))

model_highlight = MultiSelect(title="Highlight Selected Models", value=[], options=models, width=200, size=6)
model_choice.js_on_change("value", CustomJS(code="""
    console.log('scenario_choice: value=' + this.value, this.toString())
"""))

model_select = Dropdown(label="Highlight model", menu=models)
model_choice.js_on_change("value", CustomJS(code="""
    console.log('scenario_choice: value=' + this.value, this.toString())
"""))

# Data manipulation for variants widgets
variants = pd.unique(alldata['Variant_Label']).tolist()
r_variant = np.unique([int(v.split('i')[0].lstrip('r')) for v in variants])
i_variant = np.unique([int(v.split('p')[0].split('i')[1]) for v in variants])
p_variant = np.unique([int(v.split('f')[0].split('p')[1]) for v in variants])
f_variant = np.unique([int(v.split('f')[1]) for v in variants if 'f' in v])

# Now set the variants widgets
r_widg = Spinner(title="Realisation", low=r_variant.min(), high=r_variant.max(), step=1, value=r_variant.max(), width=80)
r_widg.js_on_change("value", CustomJS(code="""
    console.log('fu_tol: value=' + this.value, this.toString())
"""))

i_widg = Spinner(title="Initialisation", low=i_variant.min(), high=i_variant.max(), step=1, value=i_variant.max(), width=80)
i_widg.js_on_change("value", CustomJS(code="""
    console.log('fu_tol: value=' + this.value, this.toString())
"""))

p_widg = Spinner(title="Physics", low=p_variant.min(), high=p_variant.max(), step=1, value=p_variant.max(), width=80)
p_widg.js_on_change("value", CustomJS(code="""
    console.log('fu_tol: value=' + this.value, this.toString())
"""))

f_widg = Spinner(title="Forcing", low=f_variant.min(), high=f_variant.max(), step=1, value=f_variant.max(), width=80)
f_widg.js_on_change("value", CustomJS(code="""
    console.log('fu_tol: value=' + this.value, this.toString())
"""))


def load_data(baseline_period=baseline_slider.value, future_period=future_slider.value, bltol=bl_tol.value, futol=fu_tol.value, exp_ids='all', models=None, highlight=None, r=r_widg.value, i=i_widg.value, p=p_widg.value, f=f_widg.value):
    '''
    Create a big DataFrame of all experiments, and the changes for the periods
    :param baseline_period: tuple or list of length 2 (start_yr, end_yr)
    :param future_period: tuple or list of length 2 (start_yr, end_yr)
    :param bltol: integer to specify how many missing years we accept
    :param futol: integer to specify how many missing years we accept
    :param exp_ids: either a list of exp_ids or 'all'
    :return: pandas DataFrame
    '''

    df_list = []
    ## UKCP18
    df_ukcp18_hst = get_dataset(alldata, 'UKCP18', baseline_period[0], baseline_period[1], year_tol=bltol, exp_ids=exp_ids)
    if not df_ukcp18_hst.empty:
        df_ukcp18_fut = get_dataset(alldata, 'UKCP18', future_period[0], future_period[1], year_tol=futol, exp_ids=exp_ids)
        df_ukcp18 = pd.merge(df_ukcp18_hst, df_ukcp18_fut, how='inner', on=['Institution_ID', 'Source_ID', 'Variant_Label', 'Experiment_ID'], suffixes=('_hst', '_fut'))
        df_ukcp18['Change'] = df_ukcp18['Tmean_fut'] - df_ukcp18['Tmean_hst']
        df_ukcp18['Ensemble_Name'] = 'UKCP18'
        df_list.append(df_ukcp18)

    ## CMIP5
    df_cmip5_hst = get_dataset(alldata, 'CMIP5', baseline_period[0], baseline_period[1], year_tol=bltol, exp_ids=exp_ids)
    df_cmip5_fut = get_dataset(alldata, 'CMIP5', future_period[0], future_period[1], year_tol=futol, exp_ids=exp_ids)
    df_cmip5 = pd.merge(df_cmip5_hst, df_cmip5_fut, how='inner', on=['Institution_ID', 'Source_ID', 'Variant_Label', 'Experiment_ID'], suffixes=('_hst', '_fut'))
    df_cmip5['Change'] = df_cmip5['Tmean_fut'] - df_cmip5['Tmean_hst']
    df_cmip5['Ensemble_Name'] = 'CMIP5'
    df_list.append(df_cmip5)

    ## CMIP6
    df_cmip6_hst = get_dataset(alldata, 'CMIP6', baseline_period[0], baseline_period[1], year_tol=bltol, exp_ids=exp_ids)
    df_cmip6_fut = get_dataset(alldata, 'CMIP6', future_period[0], future_period[1], year_tol=futol, exp_ids=exp_ids)
    df_cmip6 = pd.merge(df_cmip6_hst, df_cmip6_fut, how='inner', on=['Institution_ID', 'Source_ID', 'Variant_Label', 'Experiment_ID'], suffixes=('_hst', '_fut'))
    df_cmip6['Change'] = df_cmip6['Tmean_fut'] - df_cmip6['Tmean_hst']
    df_cmip6['Ensemble_Name'] = 'CMIP6'
    df_list.append(df_cmip6)

    ## Combine all together
    df_comb = pd.concat(df_list)
    df_comb.reset_index(inplace=True)
    df_comb.drop(columns='index', inplace=True)
    df_comb['Highlight'] = 'Other models'

    if models:
        df_comb = df_comb.loc[df_comb['Source_ID'].isin(models)]

    df_comb['r'] = [int(v.split('i')[0].lstrip('r')) for v in df_comb['Variant_Label'].to_list()]
    df_comb['i'] = [int(v.split('p')[0].split('i')[1]) for v in df_comb['Variant_Label'].to_list()]
    df_comb['p'] = [int(v.split('f')[0].split('p')[1]) for v in df_comb['Variant_Label'].to_list()]
    df_comb['f'] = [int(v.split('f')[1]) if 'f' in v else np.nan for v in df_comb['Variant_Label'].to_list()]
    df_comb = df_comb[df_comb['r'] <= r]
    df_comb = df_comb[df_comb['i'] <= i]
    df_comb = df_comb[df_comb['p'] <= p]
    df_comb = df_comb[pd.isna(df_comb['f']) | (df_comb['f'] <= f)]

    if highlight:
        highrows = df_comb['Source_ID'].isin(highlight)
        print(highrows)
        df_comb.loc[highrows, 'Highlight'] = df_comb.loc[highrows, 'Source_ID']
        i = df_comb[highrows].index.to_list()
        source = ColumnDataSource(df_comb)
        source.selected.indices = i
        return source
    else:
        return ColumnDataSource(df_comb)


# Load the data for the first time, using the function we've just created
source = load_data()
# Make the plot for the first time, using the data
plot, plot_source, renderer = make_plot(baseline_slider.value, future_slider.value)
data_table = printTable()


def update_plot(attrname, old, new):
    '''
    This function does all the updating on the plot that is required on user interactions. It works for all widgets
    :param attrname: Name of the attribute we're updating
    :param old: What the value was
    :param new: What the value has been chosen to become
    :return: Nothing, but it updates the live data, and updates the plot parameters
    '''
    print(f'updating {attrname} from {old} to {new}')
    src = load_data(baseline_period=baseline_slider.value, future_period=future_slider.value, bltol=bl_tol.value, futol=fu_tol.value, exp_ids=scenario_choice.value, models=model_choice.value, highlight=model_highlight.value, r=r_widg.value, i=i_widg.value, p=p_widg.value, f=f_widg.value)
    plot.x_range.factors = sorted(np.unique(src.data['Experiment_ID']))

    plot_source.data.update(src.data)
    i, = np.where(plot_source.data['Highlight'] != 'Other models')
    plot_source.selected.update(indices=i.tolist())
    print(f"plot_source.selected: {plot_source.selected.indices}")

    # Set the plot title
    if not plot_source.selected.indices:
        plot.title.text = f"Global warming levels for the period {future_slider.value[0]} to {future_slider.value[1]}, with respect to a baseline period of {baseline_slider.value[0]} to {baseline_slider.value[1]}"
    else:
        # renderer.selection_glyph.update(fill_alpha=0.8, fill_color=fc, line_color="black")
        # renderer.nonselection_glyph.update(fill_alpha=0.3, fill_color=fc, line_color=None)
        model_list = sorted(np.unique(plot_source.data['Highlight'][i]))
        plot.title.text = f"Global warming levels for the period {future_slider.value[0]} to {future_slider.value[1]}, with respect to a baseline period of {baseline_slider.value[0]} to {baseline_slider.value[1]}, highlighting {', '.join(model_list)}"

    print(f"renderer.selection_glyph: {renderer.selection_glyph.fill_alpha}")
    print(f"renderer.selection_glyph: {renderer.selection_glyph}")
    print(f"renderer.nonselection_glyph: {renderer.nonselection_glyph}")
    
    # Make sure the color map is still set (perhaps don't need this)

    # renderer.glyph.fill_color = fc  # fc = Factor cmap set at the start
    # renderer.selection_glyph.fill_color = fc
    # renderer.nonselection_glyph.fill_color = fc


# def update_table(attr, old, new):
#     print(f'updating table {attr} from {old} to {new}')
#     i = plot_source.selected.indices
#
#     if not i:
#         i_all = data_table.source.data['index'].tolist()
#         view = CDSView(source=plot_source, filters=[IndexFilter(i_all)])
#         data_table.update(view=view)
#     else:
#         view = CDSView(source=plot_source, filters=[IndexFilter(i)])
#         data_table.update(view=view)

    # data_table.source.selected.update(indices=i.tolist())

# These lines set what happens when the widgets change
baseline_slider.on_change('value', update_plot)
future_slider.on_change('value', update_plot)
scenario_choice.on_change('value', update_plot)
model_choice.on_change('value', update_plot)
model_highlight.on_change('value', update_plot)
bl_tol.on_change('value', update_plot)
fu_tol.on_change('value', update_plot)
r_widg.on_change('value', update_plot)
i_widg.on_change('value', update_plot)
p_widg.on_change('value', update_plot)
f_widg.on_change('value', update_plot)
# plot_source.on_change('selected', update_table)


# Some generic html to describe the plots
div = Div(text="""<p>This figure allows the user to view the Global Warming Level derived from models for different 
user-defined baseline and future periods. It also allows for a certain amount of missing data within those periods (in 
the "tolerance" box), which defines the number of years of missing data that is allowed within the baseline or future 
periods. The figure is based on monthly mean 2m air temperature (over both land and ocean) output from global models, 
which is further averaged to provide an annual global mean temperature. Data is then selected from this dataset, with 
the period being defined as start_year <= x < end_yr (where x is any year in the time series). 
We currently display data from CMIP5 and CMIP6, with the intention to add UKCP09 and UKCP18 global model 
ensembles in the future. </p>
<p>Created by: <a href="mailto:andrew.hartley@metoffice.gov.uk">Andy Hartley</a> and 
<a href="mailto:anna.bradley@metoffice.gov.uk">Anna Bradley</a></p>""", height=200, sizing_mode='stretch_width')

# Finally, create the layout of the page and add it to the current document
sliders = column(row(baseline_slider, bl_tol), row(future_slider, fu_tol), sizing_mode='stretch_both')
variant_selectors = column(row(r_widg, i_widg), row(p_widg, f_widg))
controls = row(sliders, row(scenario_choice, model_choice), variant_selectors)
# controls = row(sliders, row(scenario_choice, model_choice, model_highlight), variant_selectors)
curdoc().add_root(column(controls, plot, data_table, div))
curdoc().title = "Global Warming Levels for varying periods"
