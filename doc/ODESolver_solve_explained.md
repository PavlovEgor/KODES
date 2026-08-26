# Как устроено интегрирование в `Foam::ODESolver::solve`

Важно понимать главное: **`ODESolver` — абстрактный класс, у него нет собственного алгоритма интегрирования**. Он задаёт только "протокол" из трёх перегруженных `solve()`, которые вызывают друг друга по цепочке, а реальная математика (формулы Рунге-Кутты, оценка ошибки, экстраполяция и т.д.) находится в конкретных наследниках (`RKCK45`, `Rosenbrock34`, `seulex`, ...) и подключается через **виртуальный вызов**, разрывающий цепочку.

Схема уровней:

```
solve(xStart, xEnd, y, dxTry)     ["довести систему от xStart до xEnd"]
        │  цикл for, пока x < xEnd
        ▼
solve(x, y, step)                 ["сделать один допустимый подшаг"]
        │  тонкая обёртка
        ▼
solve(x, y, dxTry)                ["виртуальный" — переопределяется в наследнике]
        │
        ▼
RKCK45 / Rosenbrock34 / ... ::solve(x, y, dxTry)   ["настоящая арифметика"]
```

## Уровень 1: внешний "водитель" — от `xStart` до `xEnd`

```cpp
void Foam::ODESolver::solve
(
    const scalar xStart,
    const scalar xEnd,
    scalarField& y,
    scalar& dxTry
) const
{
    stepState step(dxTry);
    scalar x = xStart;

    for (label nStep=0; nStep<maxSteps_; ++nStep)
    {
        scalar dxTry0 = step.dxTry;
        step.reject = false;

        // Check if this is a truncated step and set dxTry to integrate to xEnd
        if ((x + step.dxTry - xEnd)*(x + step.dxTry - xStart) > 0)
        {
            step.last = true;
            step.dxTry = xEnd - x;
        }

        // Integrate as far as possible up to step.dxTry
        solve(x, y, step);

        // Check if reached xEnd
        if ((x - xEnd)*(xEnd - xStart) >= 0)
        {
            if (nStep > 0 && step.last)
            {
                step.dxTry = dxTry0;
            }
            dxTry = step.dxTry;
            return;
        }

        step.first = false;

        if (step.reject)
        {
            step.prevReject = true;
        }
    }

    FatalErrorInFunction << ... << exit(FatalError);
}
```

Что здесь происходит по шагам цикла:

1. **`dxTry0 = step.dxTry`** — запоминаем "естественный" шаг, который решатель хотел бы сделать (унаследован из предыдущего вызова, например для соседней ячейки).
2. **Проверка перелёта интервала**: `(x + step.dxTry - xEnd)*(x + step.dxTry - xStart) > 0`. Это выражение — стандартный трюк "точка вне отрезка `[xStart, xEnd]`" (работает и при `xEnd < xStart`, то есть при интегрировании "назад"). Если следующий шаг `x + step.dxTry` перелетает `xEnd` — это последний шаг, и `step.dxTry` **обрезается** ровно до `xEnd - x`, чтобы попасть точно в конец интервала.
3. **`solve(x, y, step)`** — здесь происходит реальная работа (уровень 2 ниже): `x` продвигается, `y` обновляется, `step.dxTry` получает новую оценку шага от решателя.
4. **Проверка достижения `xEnd`**: `(x - xEnd)*(xEnd - xStart) >= 0`. Если да — интегрирование завершено:
   - если это был **не первый** подшаг (`nStep > 0`) и он был **обрезанным** (`step.last`), возвращаемое `dxTry` **восстанавливается** к `dxTry0` — иначе наружу "утекла" бы искусственно маленькая величина (остаток до `xEnd`), и следующий вызов (для другой ячейки или следующего временного шага) начинал бы с неоправданно малого шага;
   - иначе используется реальная `step.dxTry`, полученная от решателя.
   - Функция `return`-ит — **при штатной работе от `xStart` до `xEnd` доходят за один вызов `solve(xStart, xEnd, ...)`**, именно поэтому в `ode.C` переменная `deltaT` (= `xEnd`) не нуждается в повторных вызовах снаружи.
5. Если `xEnd` не достигнут — `step.first = false`, и если шаг был отклонён (`step.reject`), взводится `step.prevReject` (это нужно решателям вроде `seulex`, которые меняют стратегию после отказа).
6. Если за `maxSteps_` внутренних подшагов до `xEnd` дойти не удалось — `FatalError`, а не тихий выход с недоинтегрированной системой.

## Уровень 2: `solve(x, y, step)` — тонкая обёртка

```cpp
void Foam::ODESolver::solve
(
    scalar& x,
    scalarField& y,
    stepState& step
) const
{
    scalar x0 = x;
    solve(x, y, step.dxTry);
    step.dxDid = x - x0;
}
```

Эта функция ничего не решает сама — она просто:
- запоминает `x0` до шага;
- зовёт `solve(x, y, step.dxTry)` — а вот это уже **виртуальный вызов** третьей перегрузки (`scalar& x, scalarField& y, scalar& dxTry`), и в рантайме он уйдёт в переопределение конкретного класса (`RKCK45::solve`, `Rosenbrock34::solve` и т.д.), а не зациклится сам на себя;
- после возврата вычисляет фактически пройденное расстояние `step.dxDid = x - x0`.

**Важный нюанс дизайна**: если бы какой-то новый наследник `ODESolver` не переопределил ни `solve(x,y,dxTry)`, ни `solve(x,y,step)`, эта пара функций рекурсивно звала бы друг друга до переполнения стека — цепочка разрывается только потому, что каждый конкретный решатель обязан перехватить один из двух уровней.

## Уровень 3: настоящая математика — на примере `RKCK45`

Большинство явных решателей (`RKCK45`, `RKF45`, `RKDP45`, `Euler`, `Trapezoid`, `Rosenbrock*`, `rodas*`) переопределяют `solve(x, y, dxTry)` одинаково — просто делегируют в общий миксин `adaptiveSolver.C:61-110`:

```cpp
void Foam::RKCK45::solve
(
    scalar& x,
    scalarField& y,
    scalar& dxTry
) const
{
    adaptiveSolver::solve(odes_, x, y, dxTry);
}
```

А вот в `adaptiveSolver::solve` уже находится цикл **accept/reject** с адаптацией шага:

```cpp
void Foam::adaptiveSolver::solve(...) const
{
    scalar dx = dxTry;
    scalar err = 0.0;

    odes.derivatives(x, y, dydx0_);

    do
    {
        // Solve step and provide error estimate
        err = solve(x, y, dydx0_, dx, yTemp_);   // <-- пробный шаг + ошибка

        if (err > 1)
        {
            scalar scale = max(safeScale_*pow(err, -alphaDec_), minScale_);
            dx *= scale;                          // шаг уменьшается

            if (dx < VSMALL)
                FatalErrorInFunction << "stepsize underflow" << exit(FatalError);
        }
    } while (err > 1);

    // Update the state
    x += dx;
    y = yTemp_;

    // If the error is small increase the step-size
    if (err > pow(maxScale_/safeScale_, -1.0/alphaInc_))
    {
        scalar scale = safeScale_*pow(err, -alphaInc_);
        dxTry = clamp(scale, minScale_, maxScale_)*dx;
    }
    else
    {
        dxTry = safeScale_*maxScale_*dx;
    }
}
```

Логика классического адаптивного контроля шага:
1. Пробуем шаг `dx` (изначально = `dxTry`), вызывая **чисто виртуальный** `solve(x0, y0, dydx0, dx, y)` — это уже сама формула Рунге-Кутты конкретной схемы (см. ниже).
2. Она возвращает **нормализованную ошибку** `err` — если `err > 1`, шаг отклоняется (`reject`), уменьшается по формуле `safeScale_ * err^(-alphaDec_)` (с нижним ограничением `minScale_`) и пробуется заново — это тот самый `do...while (err > 1)`.
3. Как только `err <= 1`, шаг принимается: `x += dx`, `y = yTemp_`.
4. Оценивается **следующий** пробный шаг `dxTry` — если текущая ошибка была намного меньше допуска, шаг увеличивается (до `maxScale_`), иначе оставляется близко к текущему — классическая формула PI-контроллера шага для встроенных RK-методов.

Сама формула шага — в `RKCK45.C:114-175`: это классический явный метод Cash-Karp 4(5) — шесть вычислений производных (`k2_..k6_`), сборка решения `y` по весам `b1,b3,b4,b6` и **отдельно** — оценка ошибки `err_` по разностным весам `e1,e3,e4,e5,e6` (разница между решениями 4-го и 5-го порядка). Ошибка затем нормализуется относительно допусков:

```cpp
// ODESolver.C:42-58
scalar Foam::ODESolver::normalizeError(...) const
{
    scalar maxErr = 0.0;
    forAll(err, i)
    {
        scalar tol = absTol_[i] + relTol_[i]*max(mag(y0[i]), mag(y[i]));
        maxErr = max(maxErr, mag(err[i])/tol);
    }
    return maxErr;
}
```
— то есть `err > 1` означает "хотя бы одна компонента вышла за допуск `absTol + relTol*|y|`".

## Отдельная ветка: `seulex` / `SIBS`

Не все решатели идут через `adaptiveSolver`. `seulex` (экстраполяция Bulirsch–Stoer для жёстких систем) переопределяет **не** `solve(x,y,dxTry)`, а сразу `solve(x, y, stepState&)` — и там уже реально используются поля `step.first`, `step.prevReject`, `step.last`, которые в цепочке RKCK45 фактически не используются (там `stepState` — просто контейнер для `dxTry`/`dxDid`). Это отдельный, более сложный алгоритм.

## Вывод к предыдущему вопросу

Именно эта трёхуровневая структура объясняет, почему для `ode`-решателя `deltaT`/`dt` снаружи не меняется: **уровень 1** (`solve(xStart, xEnd, ...)`) специально спроектирован так, чтобы гарантированно довести интегрирование до `xEnd` за один внешний вызов (или упасть в `FatalError`), сам управляя всеми внутренними подшагами и их адаптацией через уровни 2 и 3. `xEnd` — параметр по значению, наружу "утекает" только адаптированная оценка следующего шага через `dxTry` (то есть `subDeltaT` → `deltaTChem_[celli]`), но не сам факт "сколько времени осталось пройти".
