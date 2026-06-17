# Left Ventricle CFD Simulations  

Welcome to our repository! Here, you'll find all the resources, codes, and tutorials necessary to set up and replicate our CFD simulations of the left ventricle. Whether you're working with idealized or patient-specific geometries, our step-by-step guides will help you through every stage of the process.

## Table of Contents  
- [Codes and Tutorials](#codes-and-tutorials)  
  - [Geometry Pre-Processing](#geometry-pre-processing)  
  - [CFD with STAR-CCM+](#cfd-with-star-ccm)  
  - [CFD with Fluent](#cfd-with-fluent)  
- [How to Contribute](#how-to-contribute)  
- [License](#license)  

---

## Codes and Tutorials  

### Geometry Pre-Processing  
In this section, you’ll find MATLAB codes for generating STL file sequences that define ventricular wall motion required for the simulations.  

- **Idealized Geometry:**  
  We provide code for generating STL files based on known analytical wall motions.  

- **Patient-Specific Geometry:**  
  Our process includes extracting the left ventricle model from cardiotomography scans and generating STL files without predefined analytical expressions.  

Both workflows rely on volume variations derived from flow rate charts, ensuring accurate representation of ventricular dynamics.

### CFD with STAR-CCM+  
This tutorial walks you through loading the geometry, meshing, and setting up the simulation in STAR-CCM+ to replicate our left ventricle blood flow results. Each step is carefully detailed to ensure accuracy and reproducibility.

### CFD with Fluent  
This guide provides a comprehensive overview of how to perform left ventricle blood flow simulations in Fluent. It covers everything from importing the geometry and generating the mesh to configuring the necessary simulation parameters for replicating our findings.

---

## How to Contribute  
We welcome contributions! Feel free to open an issue for suggestions or submit a pull request to improve or expand the repository.

---

## License  
This project is licensed under the [MIT License](LICENSE). Please see the LICENSE file for details.

---

Thank you for exploring our work. We hope this repository serves as a valuable resource for your research and simulations!
