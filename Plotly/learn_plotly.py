import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

def example_plot_bar():
    """
    https://plotly.com/python-api-reference/generated/plotly.express.bar.html?highlight=bar#plotly.express.bar
    """
    

    my_dict={
        'Ratio':['0-1','0.5-1','1-1','1.5-1','2-1','0-1','0.5-1','1-1','1.5-1','2-1','0-1','0.5-1','1-1','1.5-1','2-1',],
        'AP':[0.348,0.361,0.343,0.329,0.328,0.357,0.367,0.377,0.380,0.355,0.404,0.405,0.409,0.410,0.420],
        'flag':[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
    }

    my_df=pd.DataFrame(my_dict)


    # fig = px.bar(my_df, x="Ratio", y="AP",labels="flag", text_auto=True,barmode='overlay',range_y=[0.3,0.43],color_discrete_sequence= px.colors.sequential.Plasma_r)
    print(px.colors.qualitative.Bold)
    color_list=['rgb(1,1,1)','rgb(100,1,1)','rgb(1,1,100)',]
    # fig = px.bar(my_df, x="Ratio", y="AP",labels="flag", text_auto=True,barmode='overlay',range_y=[0.3,0.43],marker_color= color_list)
    # fig = px.bar(my_df, x="Ratio", y="AP",labels="flag", text_auto=True,barmode='overlay',range_y=[0.3,0.43],colors= ['#A56CC1', '#A6ACEC', '#63F5EF'])
    # fig = px.bar(my_df, x="Ratio", y="AP",labels="flag", text_auto=True,barmode='overlay',range_y=[0.3,0.43],range_color=['#A56CC1', '#A6ACEC', '#63F5EF'] )
    # fig = px.bar(my_df, x="Ratio", y="AP",labels="flag", text_auto=True,barmode='overlay',range_y=[0.3,0.43],color=['#A56CC1','#A56CC1','#A56CC1','#A56CC1','#A56CC1', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#63F5EF','#63F5EF','#63F5EF','#63F5EF','#63F5EF'] )
    # fig = px.bar(my_df, x="Ratio", y="AP", text_auto=True,barmode='overlay',range_y=[0.3,0.43],color=['#A56CC1','#A56CC1','#A56CC1','#A56CC1','#A56CC1', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#63F5EF','#63F5EF','#63F5EF','#63F5EF','#63F5EF'] )
    fig = px.bar(my_df, x="Ratio", y="AP", text_auto=True,barmode='overlay',range_y=[0.3,0.43],color='flag' )
    # fig = px.bar(my_df, x="Ratio", y="AP",labels="flag", text_auto=True,barmode='overlay',range_y=[0.3,0.43])

    #reference https://plotly.com/python/figure-labels/

    # fig=go.Figure(
    #     data=[go.Bar(my_df, x="Ratio", y="AP",labels="flag", text_auto=True,barmode='overlay',range_y=[0.3,0.43],marker_color= ['#A56CC1', '#A6ACEC', '#63F5EF'])]
    # )

    # fig=go.Figure(
    #     data=[go.Bar(x=my_df["Ratio"], y=my_df["AP"],marker_color= ['#A56CC1','#A56CC1','#A56CC1','#A56CC1','#A56CC1', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#63F5EF','#63F5EF','#63F5EF','#63F5EF','#63F5EF'])]
    # )

    # fig=go.Figure(
    #     data=[
    #         go.Bar(x=my_df["Ratio"][0:5], y=my_df["AP"][0:5],marker_color= ['#A56CC1','#A56CC1','#A56CC1','#A56CC1','#A56CC1', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#63F5EF','#63F5EF','#63F5EF','#63F5EF','#63F5EF'][0:5],),
    #         go.Bar(x=my_df["Ratio"][5:10], y=my_df["AP"][5:10],marker_color= ['#A56CC1','#A56CC1','#A56CC1','#A56CC1','#A56CC1', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#63F5EF','#63F5EF','#63F5EF','#63F5EF','#63F5EF'][5:10],),
    #         go.Bar(x=my_df["Ratio"][10:15], y=my_df["AP"][10:15],marker_color= ['#A56CC1','#A56CC1','#A56CC1','#A56CC1','#A56CC1', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#A6ACEC', '#63F5EF','#63F5EF','#63F5EF','#63F5EF','#63F5EF'][10:15],)
    #     ]
    # )


    fig.update_layout(
    title="Plot Title",
    font=dict(
        family="Courier New, monospace",
        size=30,
        color="RebeccaPurple"
    ))
    # fig.update_layout(barmode='overlay')
    
    fig.show()



if __name__ == "__main__":
    example_plot_bar()
    

    