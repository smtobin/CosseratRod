#ifndef __CROSS_SECTION_HPP
#define __CROSS_SECTION_HPP

#include "common.hpp"

class CrossSection
{
public:
    CrossSection(double rx, double ry)
        : _rx(rx), _ry(ry)
    {}

    virtual std::string type() const = 0;

    Real rx() const { return _rx; }
    Real ry() const { return _ry; }
    Real Ix() const { return _Ix; }
    Real Iy() const { return _Iy; }
    Real Iz() const { return _Iz; }
    Real Qxy() const { return _Qxy; }
    Real Qx() const { return _Qx; }
    Real Qy() const { return _Qy; }
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
    Real _Qxy; // area integral of x^2*y^2 over the cross-section
    Real _Qx;   // area integral of y^4 over the cross-section
    Real _Qy;   // area integral of x^4 over the cross-section
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

        // TODO: compute _Ix2y2, _Ix2, _Iy2 for elliptical cross sections

        // find the torsionally corrected polar moment of inertia
        // from https://en.wikipedia.org/wiki/Torsion_constant
        Real a = std::max(rx, ry);
        Real b = std::min(rx, ry);
        Real torsionally_corrected_Iz = (M_PI * a*a*a * b*b*b) / (a*a + b*b);

        _torsional_correction = torsionally_corrected_Iz / _Iz;
    }

    virtual std::string type() const override { return "Ellipse"; }
};

class RectCrossSection : public CrossSection
{
public:
    RectCrossSection(Real sx, Real sy)
        : CrossSection(0.5*sx, 0.5*sy)
    {
        _Ix = sx * sy * sy * sy / 12.0;
        _Iy = sy * sx * sx * sx / 12.0;
        _Iz = _Ix + _Iy;
        _A0 = sx*sy;

        _Qxy = sx*sx*sx / 12 * sy*sy*sy / 12; // area integral of x^2 y^2
        _Qx = sx * sy*sy*sy*sy*sy / 80;    // area integral of y^4
        _Qy = sy * sx*sx*sx*sx*sx / 80;    // area integral of x^4

        // find the torsionally corrected polar moment of inertia
        // from https://en.wikipedia.org/wiki/Torsion_constant
        Real torsionally_corrected_Iz;
        Real a = std::max(sx,sy);
        Real b = std::min(sx,sy);
        if (a/b >= 20.0)
            torsionally_corrected_Iz = 0.333 * a * b*b*b;
        else if (a/b >= 10.0)
            torsionally_corrected_Iz = 0.312 * a * b*b*b;
        else if (a/b >= 6.0)
            torsionally_corrected_Iz = 0.299 * a * b*b*b;
        else if (a/b >= 5.0)
            torsionally_corrected_Iz = 0.291 * a * b*b*b;
        else if (a/b >= 4.0)
            torsionally_corrected_Iz = 0.281 * a * b*b*b;
        else if (a/b >= 3.0)
            torsionally_corrected_Iz = 0.263 * a * b*b*b;
        else if (a/b >= 2.5)
            torsionally_corrected_Iz = 0.249 * a * b*b*b;
        else if (a/b >= 2.0)
            torsionally_corrected_Iz = 0.229 * a * b*b*b;
        else if (a/b >= 1.5)
            torsionally_corrected_Iz = 0.196 * a * b*b*b;
        else if (a/b >= 1.0)
            torsionally_corrected_Iz = 0.141 * a * b*b*b;
        else
            assert(0); // shouldn't get to here - a/b should always be >= 1

        _torsional_correction = torsionally_corrected_Iz / _Iz;
    }

    virtual std::string type() const override { return "Rect"; }
};

#endif // __CROSS_SECTION_HPP