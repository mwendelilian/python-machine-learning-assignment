import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#Prepare the training data
x=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
# label the data
y=[30,40,45,55,65,70,80,85,90,95]
#create a model
model=LinearRegression()
#train the model
model.fit(x,y)
#prediction
#Allow the user to enter the number of hours studied using input
hours_studied = float(input("Enter the number of hours studied: "))
#Use predict() to estimate the examination score
predicted = model.predict([[hours_studied]])
print(round(predicted[0],2))
# Create the graph
plt.scatter(x, y, color="red", label="Actual exam scores")
plt.plot(x, model.predict(x), color="blue", label="Regression line")

# Add graph labels and title
plt.xlabel("Hours Studied")
plt.ylabel("Examination Score")
plt.title("Study Hours vs Examination Scores")
plt.legend()
plt.grid(True)

# Display the graph
plt.show()
