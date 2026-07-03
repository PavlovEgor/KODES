/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
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

#include "makeKodesChemistrySolverTypes.H"

#include "thermoPhysicsTypes.H"
#include "psiReactionThermo.H"
#include "rhoReactionThermo.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    // Chemistry solvers based on sensibleEnthalpy
    makeKodesChemistrySolverTypes(psiReactionThermo, constGasHThermoPhysics);
    makeKodesChemistrySolverTypes(psiReactionThermo, gasHThermoPhysics);
    makeKodesChemistrySolverTypes(psiReactionThermo, PengRobinsonGasHThermoPhysics);
    makeKodesChemistrySolverTypes
    (
        psiReactionThermo,
        constIncompressibleGasHThermoPhysics
    );
    makeKodesChemistrySolverTypes
    (
        psiReactionThermo,
        incompressibleGasHThermoPhysics
    )
    ;
    makeKodesChemistrySolverTypes(psiReactionThermo, icoPoly8HThermoPhysics);
    makeKodesChemistrySolverTypes(psiReactionThermo, constFluidHThermoPhysics);
    makeKodesChemistrySolverTypes
    (
        psiReactionThermo,
        constAdiabaticFluidHThermoPhysics
    );
    makeKodesChemistrySolverTypes(psiReactionThermo, constHThermoPhysics);


    makeKodesChemistrySolverTypes(rhoReactionThermo, constGasHThermoPhysics);
    makeKodesChemistrySolverTypes(rhoReactionThermo, gasHThermoPhysics);
    makeKodesChemistrySolverTypes(rhoReactionThermo, PengRobinsonGasHThermoPhysics);
    makeKodesChemistrySolverTypes
    (
        rhoReactionThermo,
        constIncompressibleGasHThermoPhysics
    );
    makeKodesChemistrySolverTypes
    (
        rhoReactionThermo,
        incompressibleGasHThermoPhysics
    );
    makeKodesChemistrySolverTypes(rhoReactionThermo, icoPoly8HThermoPhysics);
    makeKodesChemistrySolverTypes(rhoReactionThermo, constFluidHThermoPhysics);
    makeKodesChemistrySolverTypes
    (
        rhoReactionThermo,
        constAdiabaticFluidHThermoPhysics
    );
    makeKodesChemistrySolverTypes(rhoReactionThermo, constHThermoPhysics);


    // Chemistry solvers based on sensibleInternalEnergy
    makeKodesChemistrySolverTypes(psiReactionThermo, constGasEThermoPhysics);
    makeKodesChemistrySolverTypes(psiReactionThermo, gasEThermoPhysics);
    makeKodesChemistrySolverTypes(psiReactionThermo, PengRobinsonGasEThermoPhysics);
    makeKodesChemistrySolverTypes
    (
        psiReactionThermo,
        constIncompressibleGasEThermoPhysics
    );
    makeKodesChemistrySolverTypes
    (
        psiReactionThermo,
        incompressibleGasEThermoPhysics
    );
    makeKodesChemistrySolverTypes(psiReactionThermo, icoPoly8EThermoPhysics);
    makeKodesChemistrySolverTypes(psiReactionThermo, constFluidEThermoPhysics);
    makeKodesChemistrySolverTypes
    (
        psiReactionThermo,
        constAdiabaticFluidEThermoPhysics
    );
    makeKodesChemistrySolverTypes(psiReactionThermo, constEThermoPhysics);

    makeKodesChemistrySolverTypes(rhoReactionThermo, constGasEThermoPhysics);
    makeKodesChemistrySolverTypes(rhoReactionThermo, gasEThermoPhysics);
    makeKodesChemistrySolverTypes(rhoReactionThermo, PengRobinsonGasEThermoPhysics);
    makeKodesChemistrySolverTypes
    (
        rhoReactionThermo,
        constIncompressibleGasEThermoPhysics
    );
    makeKodesChemistrySolverTypes
    (
        rhoReactionThermo,
        incompressibleGasEThermoPhysics
    );
    makeKodesChemistrySolverTypes(rhoReactionThermo, icoPoly8EThermoPhysics);

    makeKodesChemistrySolverTypes(rhoReactionThermo, constFluidEThermoPhysics);
    makeKodesChemistrySolverTypes
    (
        rhoReactionThermo,
        constAdiabaticFluidEThermoPhysics
    );
    makeKodesChemistrySolverTypes(rhoReactionThermo, constEThermoPhysics);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
