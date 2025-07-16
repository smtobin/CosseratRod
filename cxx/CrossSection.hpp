#ifndef __CROSS_SECTION_HPP
#define __CROSS_SECTION_HPP

#include "common.hpp"

class CrossSection
{
public:
    CrossSection(double rx, double ry)
        : _rx(rx), _ry(ry)
    {}

    Real rx() const { return _rx; }
    Real ry() const { return _ry; }
    Real Ix() const { return _Ix; }
    Real Iy() const { return _Iy; }
    Real Iz() const { return _Iz; }
    Real A0() const { return _A0; }
    Real torsionalCorrection() const { return _torsional_correction; }

    Real deformedArea(const Mat3r& C) const
    {
        return C.determinant() * _A0;
    }

    protected:
    Real _rx;   // radius in x direction
    Real _ry;   // radius in y direction
    Real _Ix;   // bending moment about x-axis
    Real _Iy;   // bending moment about y-axis
    Real _Iz;   // bending moment about z-axis
    Real _A0;   // initial (undeformed) area
    Real _torsional_correction; // torsional correction factor (for non-circular cross-sections)
};

class EllipseCrossSection : public CrossSection
{
public:
    EllipseCrossSection(Real rx, Real ry)
        : CrossSection(rx, ry)
    {
        _Ix = 0.25 * M_PI * _rx * _ry * _ry * _ry;
        _Iy = 0.25 * M_PI * _ry * _rx * _rx * _rx;
        _Iz = _Ix + _Iy;
        _A0 = M_PI * _rx * _ry;

        _torsional_correction = 1;  // TODO: update for ellipses
    }
};

class RectCrossSection : public CrossSection
{
public:
    RectCrossSection(Real sx, Real sy)
        : CrossSection(0.5*sx, 0.5*sy)
    {
        _Ix = sx * sy * sy * sy / 12.0;
        _Iy = sy * sx * sx * sx / 12.0;
        _A0 = sx*sy;
    }
};

#endif // __CROSS_SECTION_HPP