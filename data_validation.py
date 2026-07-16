import pandas as pd
import re

def validate_customer_data(data):
    errors = []
    warnings = []
    
    if not isinstance(data, dict):
        errors.append("数据格式错误：必须为字典类型")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    required_fields = ["name", "age", "month_income", "debt", "loan_count"]
    for field in required_fields:
        if field not in data:
            errors.append(f"必填字段缺失：{field}")
    
    if "name" in data:
        if not isinstance(data["name"], str) or len(data["name"].strip()) == 0:
            errors.append("姓名格式错误：必须为非空字符串")
    
    if "age" in data:
        if not isinstance(data["age"], (int, float)):
            errors.append("年龄格式错误：必须为数字")
        elif data["age"] < 18 or data["age"] > 100:
            errors.append("年龄范围错误：必须在18-100岁之间")
    
    if "month_income" in data:
        if not isinstance(data["month_income"], (int, float)):
            errors.append("月收入格式错误：必须为数字")
        elif data["month_income"] <= 0:
            errors.append("月收入必须大于0")
        elif data["month_income"] > 1000000:
            warnings.append("月收入异常：超过100万，请确认数据真实性")
    
    if "debt" in data:
        if not isinstance(data["debt"], (int, float)):
            errors.append("负债格式错误：必须为数字")
        elif data["debt"] < 0:
            errors.append("负债不能为负数")
    
    if "loan_count" in data:
        if not isinstance(data["loan_count"], int):
            errors.append("贷款次数格式错误：必须为整数")
        elif data["loan_count"] < 0:
            errors.append("贷款次数不能为负数")
        elif data["loan_count"] > 50:
            warnings.append("贷款次数异常：超过50次，请确认数据真实性")
    
    if "phone" in data and data["phone"]:
        if not re.match(r'^1[3-9]\d{9}$', str(data["phone"])):
            errors.append("手机号格式错误：必须为11位数字")
    
    if "email" in data and data["email"]:
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', data["email"]):
            errors.append("邮箱格式错误")
    
    if "month_income" in data and "debt" in data:
        if data["month_income"] > 0:
            debt_ratio = data["debt"] / data["month_income"]
            if debt_ratio > 3:
                warnings.append(f"资产负债率过高：{debt_ratio:.2f}，建议关注还款能力")
            elif debt_ratio > 1:
                warnings.append(f"资产负债率偏高：{debt_ratio:.2f}")
    
    if "id_card" in data and data["id_card"]:
        id_card = str(data["id_card"])
        if len(id_card) != 18:
            errors.append("身份证号格式错误：必须为18位")
        else:
            if not id_card[:-1].isdigit() or not (id_card[-1].isdigit() or id_card[-1].upper() == 'X'):
                errors.append("身份证号格式错误：前17位必须为数字，最后一位为数字或X")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total_fields": len(data),
        "valid_fields": len(data) - len(errors)
    }

def validate_batch_data(df):
    validation_results = []
    total_errors = 0
    total_warnings = 0
    valid_records = 0
    
    for idx, row in df.iterrows():
        result = validate_customer_data(row.to_dict())
        validation_results.append({
            "record_id": idx + 1,
            **result
        })
        if result["valid"]:
            valid_records += 1
        total_errors += len(result["errors"])
        total_warnings += len(result["warnings"])
    
    return {
        "summary": {
            "total_records": len(df),
            "valid_records": valid_records,
            "invalid_records": len(df) - valid_records,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "validation_rate": valid_records / len(df) * 100
        },
        "details": validation_results
    }

def generate_validation_report(batch_result):
    summary = batch_result["summary"]
    report = f"""
数据校验报告
================================

一、校验概况
总记录数：{summary['total_records']}
有效记录数：{summary['valid_records']}
无效记录数：{summary['invalid_records']}
校验通过率：{summary['validation_rate']:.2f}%
总错误数：{summary['total_errors']}
总警告数：{summary['total_warnings']}

二、校验结论
{'✓ 数据质量良好，校验通过' if summary['validation_rate'] >= 95 else 
  '⚠ 数据存在部分问题，建议核查' if summary['validation_rate'] >= 80 else 
  '✗ 数据质量较差，需全面检查'}

三、处理建议
1. 对无效记录进行人工复核修正
2. 关注警告记录中的异常数据
3. 建议在数据采集环节增加前端校验
"""
    return report

if __name__ == "__main__":
    test_data = {
        "name": "张三",
        "age": 28,
        "month_income": 15000,
        "debt": 5000,
        "loan_count": 3,
        "phone": "13812345678",
        "email": "zhangsan@example.com"
    }
    
    result = validate_customer_data(test_data)
    print("单条数据校验结果：")
    print(f"有效：{result['valid']}")
    print(f"错误：{result['errors']}")
    print(f"警告：{result['warnings']}")
    
    df = pd.DataFrame([
        test_data,
        {"name": "", "age": 15, "month_income": -1000, "debt": 10000, "loan_count": 5},
        {"name": "李四", "age": 35, "month_income": 8000, "debt": 20000, "loan_count": 8}
    ])
    
    batch_result = validate_batch_data(df)
    print("\n批量数据校验报告：")
    print(generate_validation_report(batch_result))