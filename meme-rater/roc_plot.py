import matplotlib.pyplot as plt
import json

data = json.loads("[[1.2792096138000488,true],[1.1153279542922974,true],[0.9720794558525085,true],[-0.5180545449256897,false],[1.4547114372253418,true],[1.3289614915847778,true],[1.8748269081115723,true],[0.05465051531791687,false],[0.7888763546943665,true],[1.368210792541504,true],[1.4808461666107178,true],[0.9501181244850159,true],[1.2592355012893677,true],[1.0127032995224,true],[-0.8805797100067139,false],[-0.08946493268013,true],[0.4224545955657959,false],[1.0051900148391724,true],[0.5121232271194458,false],[1.0876282453536987,false],[1.5552432537078857,true],[-0.3680466413497925,false],[0.45498305559158325,true],[1.3851803541183472,true],[-0.8842921853065491,false],[2.6869430541992188,false],[1.6892706155776978,false],[0.7087478637695312,false],[-0.5138207077980042,false],[0.16498255729675293,false],[1.265992283821106,true],[0.47311416268348694,false],[0.04918492212891579,false],[1.283980369567871,true],[1.0510015487670898,false],[1.6323922872543335,false],[0.4570896625518799,true],[1.5262614488601685,true],[1.4057230949401855,true],[1.0391144752502441,true],[0.9190238118171692,true],[1.2970502376556396,true],[2.025949478149414,true],[0.6396026611328125,true],[2.3505871295928955,true],[1.0854156017303467,false],[1.0216373205184937,true],[-1.163207769393921,false],[1.8854788541793823,true],[0.249663308262825,false],[-0.8619526028633118,false],[1.9995672702789307,true],[1.0939114093780518,false],[0.6106101870536804,false],[1.8383781909942627,false],[-0.0637127161026001,false],[-0.34953051805496216,false],[0.988452672958374,false],[0.5209289193153381,false],[-0.4708566963672638,false],[0.4715256690979004,false],[-0.7905446887016296,false],[2.0255637168884277,true],[0.8488644361495972,false],[1.6645262241363525,true],[1.0948383808135986,true],[-0.8315924406051636,false],[1.5533114671707153,true],[0.9333463907241821,true],[-0.5723654627799988,false],[1.9510998725891113,true],[0.2842162549495697,false],[1.1901239156723022,false],[1.5058742761611938,false],[0.7622374296188354,false],[0.2894713282585144,false],[0.0965774804353714,false],[0.6335093379020691,false],[-0.7369110584259033,false],[1.2673722505569458,true],[0.9775630235671997,false],[0.7889275550842285,false],[-0.9432369470596313,false],[0.24122865498065948,false],[1.075297474861145,false],[0.545269250869751,false],[-0.1398508995771408,false],[-0.31118375062942505,false],[1.47971510887146,false],[0.5115379691123962,true],[0.8894630074501038,true],[0.4365079700946808,true],[2.5944597721099854,true],[0.8613907694816589,false],[1.1540073156356812,false],[1.6798168420791626,true],[1.5266021490097046,true],[0.2556634545326233,false],[0.90388423204422,false],[0.36393579840660095,false],[1.297504186630249,true],[1.091887354850769,true],[0.931088924407959,true],[0.8854649066925049,true],[0.0385725162923336,false],[1.5259686708450317,true],[-0.725635826587677,false],[-1.72086501121521,false],[1.9044498205184937,true],[-0.10369344800710678,false],[-0.5889104604721069,true],[0.2478746473789215,false],[1.4628609418869019,false],[1.1434470415115356,false],[0.20635242760181427,false],[0.8324120044708252,false],[0.676543653011322,false],[1.1111537218093872,true],[0.0488731786608696,false],[0.8705015182495117,true],[0.5464357733726501,true],[0.6190940737724304,true],[0.33756133913993835,false],[0.8019527196884155,true],[1.1540179252624512,true],[-1.4343260526657104,true],[1.4069069623947144,true],[0.5078597664833069,true],[0.1831521838903427,false],[-0.5352457761764526,false],[1.3706591129302979,true],[-0.8636290431022644,false],[0.8164027333259583,false],[0.6665022969245911,false],[0.5028047561645508,false],[-0.7765756845474243,false],[1.204775333404541,false],[1.2527906894683838,false],[0.7420544028282166,false],[1.0363034009933472,true],[1.0559784173965454,false],[-0.72457355260849,false],[1.9217685461044312,true],[0.9770780205726624,false],[0.8808136582374573,true],[1.0174754858016968,false],[0.4287119507789612,false],[1.0718724727630615,true],[0.8409612774848938,true],[-1.3366127014160156,false]]")
data = sorted(data, reverse=True)

tprs, fprs = [], []
positives = sum(1 for _, ground_truth in data if ground_truth)
negatives = len(data) - positives

for threshold, _ in data:
    tp = sum(1 for score, ground_truth in data if ground_truth and score >= threshold)
    fp = sum(1 for score, ground_truth in data if not ground_truth and score >= threshold)
    tpr = tp / positives
    fpr = fp / negatives
    tprs.append(tpr)
    fprs.append(fpr)

auroc = 0
for i in range(len(fprs) - 1):
    auroc += (fprs[i+1] - fprs[i]) * (tprs[i+1] + tprs[i]) / 2

print(f"AUROC: {auroc}")

plt.plot(fprs, tprs)

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC")

plt.tight_layout()
plt.show()
