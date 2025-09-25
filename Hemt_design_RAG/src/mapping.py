import json


material_mapping = {
    "GaN": ["氮化镓","氮化镓衬底", "GaN", "gan", "Gan","GAN","GaN衬底","未掺杂GaN（Undoped GaN）"],
    "Si":["硅","硅衬底","si","SI","Si","Si衬底"],
    "Sapphire":["蓝宝石","sapphire","sapphire substrate","蓝宝石衬底","蓝宝石（Sapphire）"],
    "AlN":["氮化铝","铝氮化物","aluminum nitride"],
    "diamond":["金刚石","金刚石衬底","diamond","diamond substrate"],
    "SiN":["氮化硅","silicon nitride","SiN","硅氮化物"],
    "AlGaN":["AlGaN barrier","AlGaN","铝镓氮","Algan","AlGan","AlGaN","AlGAN"]
}

if __name__ == "__main__":
    out_json = "D:/Prompts/8.7-8.11-合并json-更新Prompt-提交有问题的条目/3000_补交+10442/out.jsonl"
    output = "D:/deep_thought/out.jsonl"
    out = open(output,'w',encoding="utf-8")
    f = open(out_json,'r',encoding="utf-8")
    for file in f:
        data = json.loads(file)
        answer = data["answer"]
        new_answer = answer
        for norlized_name, material_names in material_mapping.items(): 
            for material in material_names:
                new_answer = new_answer.replace(material,norlized_name)
        paper = data["paper"]
        mineruID = data["mineruID"]
        result = {
            "answer":new_answer,
            "paper":paper,
            "mineruID":mineruID
        }
        out.write(json.dumps(result,ensure_ascii=False)+'\n')
        