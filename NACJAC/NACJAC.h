
// NACJAC.h : main header file for the PROJECT_NAME application
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols

#define TITLE L"Neo Armstrong Cyclone Jet Armstrong Bruteforcer"

// CNACJACApp:
// See NACJAC.cpp for the implementation of this class
//

class CNACJACApp : public CWinApp
{
public:
	CNACJACApp();

// Overrides
public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern CNACJACApp theApp;
