#!/usr/bin/env python3
"""
Анализ времени расчета reactingFoam
Время реакции = ExecutionTime_after_reaction - ExecutionTime_before_reaction
Время итерации = ExecutionTime_end - ExecutionTime_before_reaction
"""

import re
import sys

def parse_log_file(filename):
    """
    Парсит лог файл и извлекает времена для каждой PIMPLE итерации
    """
    
    # Регулярные выражения
    time_pattern = re.compile(r'^Time = ([\d\.e\-+]+)')
    pimple_pattern = re.compile(r'PIMPLE: iteration (\d+)')
    before_reaction_pattern = re.compile(r'Before reaction correct:')
    after_reaction_pattern = re.compile(r'After reaction correct:')
    execution_time_pattern = re.compile(r'ExecutionTime = ([\d\.]+) s')
    
    data = []  # Список для хранения данных по каждой итерации
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Ищем начало: Time = ...
        time_match = time_pattern.search(line)
        if time_match:
            current_time = float(time_match.group(1))
            
            # Ищем PIMPLE iteration в следующих строках
            pimple_iter = None
            for j in range(i+1, min(i+10, len(lines))):
                pimple_match = pimple_pattern.search(lines[j])
                if pimple_match:
                    pimple_iter = int(pimple_match.group(1))
                    break
            
            if pimple_iter is not None:
                # Ищем "Before reaction correct:" и ExecutionTime после него
                before_time = None
                after_time = None
                end_time = None
                
                for j in range(i, min(i+500, len(lines))):
                    line_j = lines[j].strip()
                    
                    # Ищем Before reaction correct
                    if before_reaction_pattern.search(line_j):
                        # Ищем ExecutionTime в следующих строках
                        for k in range(j+1, min(j+5, len(lines))):
                            exec_match = execution_time_pattern.search(lines[k])
                            if exec_match:
                                before_time = float(exec_match.group(1))
                                break
                    
                    # Ищем After reaction correct
                    if after_reaction_pattern.search(line_j) and before_time is not None:
                        # Ищем ExecutionTime в следующих строках
                        for k in range(j+1, min(j+5, len(lines))):
                            exec_match = execution_time_pattern.search(lines[k])
                            if exec_match:
                                after_time = float(exec_match.group(1))
                                break
                    
                    # Ищем ExecutionTime в конце итерации (после After reaction)
                    if after_time is not None and end_time is None:
                        exec_match = execution_time_pattern.search(line_j)
                        if exec_match:
                            # Берем последний ExecutionTime в этой итерации
                            current_exec = float(exec_match.group(1))
                            # Проверяем, что это не время после реакции (оно уже записано)
                            if abs(current_exec - after_time) > 0.01:  # Если отличается от after_time
                                end_time = current_exec
                    
                    # Если нашли все три времени - выходим
                    if before_time is not None and after_time is not None and end_time is not None:
                        break
                
                # Если нашли все три времени - сохраняем данные
                if before_time is not None and after_time is not None and end_time is not None:
                    reaction_time = after_time - before_time
                    iter_time = end_time - before_time
                    
                    data.append({
                        'time_step': current_time,
                        'pimple_iter': pimple_iter,
                        'before_time': before_time,
                        'after_time': after_time,
                        'end_time': end_time,
                        'reaction_time': reaction_time,
                        'iter_time': iter_time,
                        'other_time': iter_time - reaction_time
                    })
        
        i += 1
    
    return data

def analyze_and_print(data):
    """
    Анализирует и выводит статистику
    """
    
    print("=" * 80)
    print("АНАЛИЗ ВРЕМЕНИ РАСЧЕТА reactingFoam")
    print("=" * 80)
    
    if not data:
        print("Не найдено данных! Проверьте формат лог-файла.")
        print("\nУбедитесь, что в логе есть последовательность:")
        print("  Time = ...")
        print("  PIMPLE: iteration ...")
        print("  Before reaction correct:")
        print("  ExecutionTime = ...")
        print("  After reaction correct:")
        print("  ExecutionTime = ...")
        print("  ... (решения уравнений) ...")
        print("  ExecutionTime = ...")
        return
    
    print(f"\nНайдено {len(data)} PIMPLE итераций\n")
    
    # Таблица с деталями по каждой итерации
    print("{:>12} {:>10} {:>12} {:>12} {:>12} {:>12}".format(
        "Time", "PIMPLE", "Iter(s)", "Reaction(s)", "Other(s)", "Reaction %"
    ))
    print("-" * 80)
    
    reaction_times = []
    iter_times = []
    other_times = []
    
    for d in data:
        reaction_percent = (d['reaction_time'] / d['iter_time'] * 100) if d['iter_time'] > 0 else 0
        
        print("{:>12.6e} {:>10} {:>12.3f} {:>12.3f} {:>12.3f} {:>11.1f}%".format(
            d['time_step'], d['pimple_iter'],
            d['iter_time'], d['reaction_time'], d['other_time'],
            reaction_percent
        ))
        
        reaction_times.append(d['reaction_time'])
        iter_times.append(d['iter_time'])
        other_times.append(d['other_time'])
    
    # Статистика
    print("\n" + "=" * 80)
    print("СТАТИСТИКА")
    print("=" * 80)
    
    total_iter_time = sum(iter_times)
    total_reaction_time = sum(reaction_times)
    total_other_time = sum(other_times)
    
    print(f"Всего PIMPLE итераций: {len(data)}")
    print(f"\nОбщее время итераций: {total_iter_time:.2f} s")
    print(f"  - Время на химию: {total_reaction_time:.2f} s ({total_reaction_time/total_iter_time*100:.1f}%)")
    print(f"  - Остальное время: {total_other_time:.2f} s ({total_other_time/total_iter_time*100:.1f}%)")
    
    print(f"\nСреднее время итерации: {sum(iter_times)/len(iter_times):.3f} s")
    print(f"  - Среднее время химии: {sum(reaction_times)/len(reaction_times):.3f} s")
    print(f"  - Среднее остальное время: {sum(other_times)/len(other_times):.3f} s")
    
    print(f"\nМаксимальное время химии: {max(reaction_times):.3f} s")
    print(f"Минимальное время химии: {min(reaction_times):.3f} s")
    
    # Анализ по временным шагам
    print("\n" + "=" * 80)
    print("АНАЛИЗ ПО ВРЕМЕННЫМ ШАГАМ")
    print("=" * 80)
    
    time_steps = {}
    for d in data:
        ts = d['time_step']
        if ts not in time_steps:
            time_steps[ts] = {'iter_time': 0, 'reaction_time': 0, 'count': 0}
        
        time_steps[ts]['iter_time'] += d['iter_time']
        time_steps[ts]['reaction_time'] += d['reaction_time']
        time_steps[ts]['count'] += 1
    
    print("\n{:>12} {:>10} {:>15} {:>15} {:>15}".format(
        "Time", "Iterations", "Total time(s)", "Reaction(s)", "Reaction %"
    ))
    print("-" * 80)
    
    for ts in sorted(time_steps.keys()):
        ts_data = time_steps[ts]
        reaction_percent = (ts_data['reaction_time'] / ts_data['iter_time'] * 100) if ts_data['iter_time'] > 0 else 0
        
        print("{:>12.6e} {:>10} {:>15.3f} {:>15.3f} {:>14.1f}%".format(
            ts, ts_data['count'], ts_data['iter_time'], 
            ts_data['reaction_time'], reaction_percent
        ))
    
    # Сохранение в CSV
    import csv
    
    with open('pimple_analysis.csv', 'w', newline='') as csvfile:
        fieldnames = ['time_step', 'pimple_iter', 'iter_time_s', 'reaction_time_s', 
                     'other_time_s', 'reaction_percent', 'before_time', 'after_time', 'end_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for d in data:
            writer.writerow({
                'time_step': d['time_step'],
                'pimple_iter': d['pimple_iter'],
                'iter_time_s': d['iter_time'],
                'reaction_time_s': d['reaction_time'],
                'other_time_s': d['other_time'],
                'reaction_percent': d['reaction_time']/d['iter_time']*100 if d['iter_time']>0 else 0,
                'before_time': d['before_time'],
                'after_time': d['after_time'],
                'end_time': d['end_time']
            })
    
    print("\n" + "=" * 80)
    print(f"Данные сохранены в файл: pimple_analysis.csv")
    print("=" * 80)

def main():
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "log.reactingKodesFoam"
    
    try:
        print(f"Чтение файла: {log_file}\n")
        data = parse_log_file(log_file)
        analyze_and_print(data)
        
    except FileNotFoundError:
        print(f"Ошибка: Файл '{log_file}' не найден!")
        print(f"Использование: python {sys.argv[0]} <log_file>")
        sys.exit(1)

if __name__ == "__main__":
    main()