import lifelines
from dataset import FedTcgaBrca

penalizer = 0.1
model = lifelines.CoxPHFitter(penalizer=penalizer)

train = FedTcgaBrca(train=True, pooled=True).data
train = train.drop("pid", 1)
model.fit(train, "T", "E")

test = FedTcgaBrca(train=False, pooled=True).data
test = test.drop("pid", 1)
pred = -model.predict_partial_hazard(test)
c_index = lifelines.utils.concordance_index(test["T"], pred, test["E"])
print("Test c-index on pooled dataset ", c_index)
