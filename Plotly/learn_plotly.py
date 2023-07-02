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


    my_dict={
        'ratio':['0-1','0.5-1','1-1','1.5-1','2-1'],
        '50':[0.348,0.361,0.343,0.329,0.328],
        '100':[0.357,0.367,0.377,0.380,0.355],
        '150':[0.404,0.405,0.409,0.410,0.420],
    }

    my_df=pd.DataFrame(my_dict)

    fig = px.bar(my_df, x="ratio", y=['150','100','50'], text_auto=True,barmode='overlay',range_y=[0.3,0.43] )



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
    

    