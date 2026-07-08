/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2013-2016 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
    Copyright (c) 2026 Egor Pavlov
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

Class
    Foam::kodesSeulex

Description
    Line-for-line copy of OpenFOAM's own "seulex" ODE solver
    ($FOAM_SRC/ODE/ODESolvers/seulex), with diagnostic logging added so its
    output can be diffed directly against the KODES GPU Seulex log
    (src/Integrators/Seulex.cu): the state vector + dx/x at the top of every
    accepted outer step, and the raw derivatives vector on every internal
    seul() call. Used to check whether the tiny time steps seen on the GPU
    are intrinsic to the Seulex algorithm on this exact chemistry (GRI-Mech),
    or specific to the KODES port.

    Select via chemistryProperties:
        odeCoeffs { solver kodesSeulex; absTol 1e-12; relTol 1e-1; }

SourceFiles
    kodesSeulex.C

\*---------------------------------------------------------------------------*/

#ifndef kodesSeulex_H
#define kodesSeulex_H

#include "ODESolver.H"
#include "scalarMatrices.H"
#include "labelField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class kodesSeulex Declaration
\*---------------------------------------------------------------------------*/

class kodesSeulex
:
    public ODESolver
{
    // Private data

        // Static constants

            static const label kMaxx_ = 12;
            static const label iMaxx_ = kMaxx_ + 1;

            static const scalar
                stepFactor1_, stepFactor2_, stepFactor3_,
                stepFactor4_, stepFactor5_,
                kFactor1_, kFactor2_;

        // Evaluated constants

            scalar jacRedo_;
            labelField nSeq_;
            scalarField cpu_;
            scalarSquareMatrix coeff_;

        // Temporary storage
        // held to avoid dynamic memory allocation between calls
        // and to transfer internal values between functions

            mutable scalar theta_;
            mutable label kTarg_;
            mutable scalarRectangularMatrix table_;

            mutable scalarField dfdx_;
            mutable scalarSquareMatrix dfdy_;
            mutable scalarSquareMatrix a_;
            mutable labelList pivotIndices_;

            // Fields space for "solve" function
            mutable scalarField dxOpt_, temp_;
            mutable scalarField y0_, ySequence_, scale_;

            // Fields used in "seul" function
            mutable scalarField dy_, yTemp_, dydx_;

            // Diagnostic step counter (purely for log readability)
            mutable label logStepCount_;


    // Private Member Functions

        //- Computes the j-th line of the extrapolation table
        bool seul
        (
            const scalar x0,
            const scalarField& y0,
            const scalar dxTot,
            const label k,
            scalarField& y,
            const scalarField& scale
        ) const;

        //- Polynomial extrpolation
        void extrapolate
        (
            const label k,
            scalarRectangularMatrix& table,
            scalarField& y
        ) const;


public:

    //- Runtime type information
    TypeName("kodesSeulex");


    // Constructors

        //- Construct from ODESystem
        kodesSeulex(const ODESystem& ode, const dictionary& dict);


    //- Destructor
    virtual ~kodesSeulex() = default;


    // Member Functions

        //- Resize the ODE solver
        virtual bool resize();

        //- Solve the ODE system and the update the state
        virtual void solve
        (
            scalar& x,
            scalarField& y,
            stepState& step
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
