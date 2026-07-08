/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2020-2023 OpenCFD Ltd.
    Copyright (c) 2026 Egor Pavlov
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "kodesChemistryModel.H"
#include "reactingMixture.H"
#include "UniformField.H"
#include "extrapolatedCalculatedFvPatchFields.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ReactionThermo, class ThermoType>
Foam::kodesChemistryModel<ReactionThermo, ThermoType>::kodesChemistryModel
(
    ReactionThermo& thermo
)
:
    StandardChemistryModel<ReactionThermo, ThermoType>(thermo),
    // host_res(numOfSystems, NSP, 1),
    // host_res_dev(host_res.numOfSystems(), host_res.sizeOfSystem(), host_res.numOfParameters()),
    // op(&host_res, &host_res_dev)
{

    // host_res.vectors[0] = this->thermo().T().prt();
    // host_res.parameters[0] = this->thermo().p().prt();

    // for (int i=1; i < NSP; ++i)
    // {
    //     host_res.vectors[i] = Y_[i].data();
    // }

    // h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    // initialize_gpu_memory(host_res.numOfSystems(), &h_mem, &d_mem);

    // res_prt = kodes::SeulexDeviceResources::create(numOfSystems, host_res.sizeOfSystem(), 1, &host_res_dev);

    // ode_prt = kodes::GRIMESHSystem::createGPU(d_mem);

    // solver = kodes::Seulex<kodes::GRIMESHSystem> (ode_prt, res_prt, host_res.numOfSystems());

    Info<< "\n========================================" << nl
        << "  kodesChemistryModel: CUSTOM MODEL LOADED!" << nl
        << "  Using custom chemistry model by user" << nl
        << "  Number of species = " << this->nSpecie_ << nl
        << "  Number of reactions = " << this->nReaction_ << nl
        << "  Thermophysical type: " << ThermoType::typeName() << nl
        << "  Time: " << this->mesh().time().timeName() << nl
        << "========================================\n" << endl;
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ReactionThermo, class ThermoType>
Foam::kodesChemistryModel<ReactionThermo, ThermoType>::~kodesChemistryModel()
{
    Info<< "kodesChemistryModel: destructor called" << endl;

    // kodes::GRIMESHSystem::destroyGPU(ode_prt);
    // kodes::SeulexDeviceResources::destroy(res_prt, &host_res_dev);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class ReactionThermo, class ThermoType>
void Foam::kodesChemistryModel<ReactionThermo, ThermoType>::solve
(
    scalarField& c,
    scalar& T,
    scalar& p,
    scalar& deltaT,
    scalar& subDeltaT
) const
{
    this->StandardChemistryModel<ReactionThermo, ThermoType>::solve
    (
        c,
        T,
        p,
        deltaT,
        subDeltaT
    );
}

template<class ReactionThermo, class ThermoType>
Foam::scalar Foam::kodesChemistryModel<ReactionThermo, ThermoType>::solve
(
    const scalar deltaT
)
{
    return this->StandardChemistryModel<ReactionThermo, ThermoType>::solve(deltaT);
}

template<class ReactionThermo, class ThermoType>
Foam::scalar Foam::kodesChemistryModel<ReactionThermo, ThermoType>::solve
(
    const scalarField& deltaT
)
{
    return this->StandardChemistryModel<ReactionThermo, ThermoType>::solve(deltaT);
}

template<class ReactionThermo, class ThermoType>
void Foam::kodesChemistryModel<ReactionThermo, ThermoType>::calculate()
{
    this->StandardChemistryModel<ReactionThermo, ThermoType>::calculate();
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

