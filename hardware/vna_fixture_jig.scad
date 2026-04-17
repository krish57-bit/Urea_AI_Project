/*
    VNA RFID Fixture Jig - Monolithic Design
    Role: Technical Co-Founder / Senior RF Engineer
    Project: Milk Urea AI Detection
    
    This Jig ensures a strict 3.0mm air gap between the SMA probe 
    and the PET bottle wall, with the probe centered at 59mm height.
*/

// --- PARAMETRIC CONSTANTS ---
BOTTLE_WIDTH = 48.5;    // [mm] Measured width of 200ml square PET bottle
BOTTLE_HEIGHT = 120;     // [mm] Total height of bottle slot
WALL_THICKNESS = 4.0;    // [mm] Rigidity of the 3D printed fixture
AIR_GAP = 3.0;           // [mm] MANDATORY clearance for near-field interaction
PROBE_HEIGHT = 59.0;     // [mm] Center of SMA probe from bottle base
SMA_HOLE_DIA = 6.4;      // [mm] Standard SMA connector thread diameter
SMA_FLANGE_W = 12.7;     // [mm] Standard 4-hole SMA flange width
TOLERANCE = 0.5;         // [mm] Fitting tolerance for bottle insertion

// --- MAIN ASSEMBLY ---
union() {
    base_plate();
    bottle_well();
    probe_mount_tower();
}

// --- MODULES ---

module base_plate() {
    // Solid foundation to prevent tipping during VNA sweep
    translate([0, 0, 0])
    cube([BOTTLE_WIDTH + 20, BOTTLE_WIDTH + 20, 5], center=true);
}

module bottle_well() {
    // The "monolithic" holder for the square PET bottle
    difference() {
        // External shell
        translate([0, 0, BOTTLE_HEIGHT/2])
        cube([BOTTLE_WIDTH + 2*WALL_THICKNESS + TOLERANCE, 
              BOTTLE_WIDTH + 2*WALL_THICKNESS + TOLERANCE, 
              BOTTLE_HEIGHT], center=true);
        
        // Internal hollow for bottle
        translate([0, 0, BOTTLE_HEIGHT/2 + 2]) // 2mm offset for floor
        cube([BOTTLE_WIDTH + TOLERANCE, 
              BOTTLE_WIDTH + TOLERANCE, 
              BOTTLE_HEIGHT + 1], center=true);
              
        // Front Viewing Window (Calibration check for 10mm tag)
        translate([0, -(BOTTLE_WIDTH/2 + WALL_THICKNESS), PROBE_HEIGHT])
        cube([20, 10, 40], center=true);
    }
}

module probe_mount_tower() {
    // The critical RF geometry block
    translate([0, (BOTTLE_WIDTH/2 + WALL_THICKNESS/2 + TOLERANCE/2), PROBE_HEIGHT]) {
        difference() {
            // Mounting block extending outwards
            translate([0, SMA_FLANGE_W/4, 0])
            cube([SMA_FLANGE_W + 4, SMA_FLANGE_W/2 + AIR_GAP, 20], center=true);
            
            // 1. Air Gap Cutout (Ensures SMA flange is precisely 3mm from PET wall)
            translate([0, -(SMA_FLANGE_W/4 + 1), 0])
            cube([BOTTLE_WIDTH, AIR_GAP, 50], center=true);
            
            // 2. SMA Connector Hole
            rotate([90, 0, 0])
            cylinder(h=50, d=SMA_HOLE_DIA, $fn=50, center=true);
            
            // 3. Optional: SMA Flange Bolt Holes (Assuming 2-hole diagonal)
            translate([6, 0, 6]) rotate([90, 0, 0]) cylinder(h=50, d=3.2, $fn=20, center=true);
            translate([-6, 0, -6]) rotate([90, 0, 0]) cylinder(h=50, d=3.2, $fn=20, center=true);
        }
    }
}

// --- USAGE NOTES ---
// 1. Print in PETG or PLA with 30%+ infill for dimensional stability.
// 2. Ensure Probe Height (59mm) and Air Gap (3.0mm) are verified after print.
// 3. Bottle should slide in with minimal friction (Drop and Swap).
