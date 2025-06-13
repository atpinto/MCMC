
        // --- Setup: Get DOM elements and initialize variables ---
        const canvas = document.getElementById('mcmcCanvas');
        const ctx = canvas.getContext('2d');
        const marginalXCanvas = document.getElementById('marginalXCanvas');
        const marginalXCtx = marginalXCanvas.getContext('2d');
        const marginalYCanvas = document.getElementById('marginalYCanvas');
        const marginalYCtx = marginalYCanvas.getContext('2d');

        const restartButton = document.getElementById('restartButton');
        const pauseButton = document.getElementById('pauseButton');
        const numPointsInput = document.getElementById('numPoints');
        const speedSlider = document.getElementById('speedSlider');
        const speedValueSpan = document.getElementById('speedValue');
        const algorithmSelector = document.getElementById('algorithmSelector');
        const descriptionP = document.getElementById('description');
        const animationKeyUl = document.getElementById('animation-key');
        const hmcControls1 = document.getElementById('hmcControlsContainer');
        const hmcControls2 = document.getElementById('hmcControlsContainer2');
        const leapfrogStepsInput = document.getElementById('leapfrogSteps');
        const epsilonInput = document.getElementById('epsilon');

        // Animation and simulation parameters
        let n_animation_steps = 500;
        let stepDelay = 50;
        const start_point = { x: -2, y: 12 };
        const proposal_std_dev = 3.0;
        const targetMean = { x: 5, y: 5 };
        const targetStdDev = { x: 2, y: 3 };
        const targetCorrelation = 0.8;
        const domain = { min: -5, max: 15 };
        const canvasSize = 400;

        // State variables
        let history = [];
        let chainPath = [];
        let currentStep = 0;
        let animationFrameId;
        let lastUpdateTime = 0;
        let isPaused = false;
        
        // Histogram parameters
        const numBins = 40;
        const binWidth = (domain.max - domain.min) / numBins;
        let yMaxCountX, yMaxCountY;

        // --- Core Math & Distribution Functions ---
        function random_normal() {
            let u = 0, v = 0;
            while (u === 0) u = Math.random();
            while (v === 0) v = Math.random();
            return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        }

        function log_pdf_bivariate_normal(x, y) {
            const { x: mu_x, y: mu_y } = targetMean;
            const { x: s_x, y: s_y } = targetStdDev;
            const rho = targetCorrelation;
            const z = ((x - mu_x) / s_x)**2 - (2 * rho * (x - mu_x) * (y - mu_y)) / (s_x * s_y) + ((y - mu_y) / s_y)**2;
            const log_denom = Math.log(2 * Math.PI * s_x * s_y * Math.sqrt(1 - rho**2));
            return -log_denom - z / (2 * (1 - rho**2));
        }

        function grad_log_pdf(q) {
            const { x: mu_x, y: mu_y } = targetMean;
            const { x: s_x, y: s_y } = targetStdDev;
            const rho = targetCorrelation;
            const s_x2 = s_x**2;
            const s_y2 = s_y**2;

            const common_factor = -1 / (1 - rho**2);
            const grad_x = common_factor * ( (q.x - mu_x) / s_x2 - (rho * (q.y - mu_y)) / (s_x * s_y) );
            const grad_y = common_factor * ( (q.y - mu_y) / s_y2 - (rho * (q.x - mu_x)) / (s_x * s_y) );
            
            return { x: grad_x, y: grad_y };
        }

        function pdf_normal(x, mean, stdDev) {
            return Math.exp(-0.5 * ((x - mean) / stdDev) ** 2) / (stdDev * Math.sqrt(2 * Math.PI));
        }

        // --- MCMC Algorithms ---
        function runMCMC(num_steps) {
            let current_point = { ...start_point };
            const fullHistory = [];
            for (let i = 0; i < num_steps; i++) {
                const jump = { x: random_normal() * proposal_std_dev, y: random_normal() * proposal_std_dev };
                const proposal_point = { x: current_point.x + jump.x, y: current_point.y + jump.y };
                const log_p_proposal = log_pdf_bivariate_normal(proposal_point.x, proposal_point.y);
                const log_p_current = log_pdf_bivariate_normal(current_point.x, current_point.y);
                const acceptance_ratio = Math.exp(log_p_proposal - log_p_current);
                
                let accepted = (Math.random() < acceptance_ratio);
                fullHistory.push({ start: { ...current_point }, proposal: proposal_point, accepted: accepted });
                if (accepted) {
                    current_point = { ...proposal_point };
                }
            }
            return fullHistory;
        }

        function runGibbs(num_steps) {
            let current_point = { ...start_point };
            const fullHistory = [];
            const { x: mu_x, y: mu_y } = targetMean;
            const { x: sigma_x, y: sigma_y } = targetStdDev;
            const rho = targetCorrelation;

            for (let i = 0; i < num_steps; i++) {
                const mean_x_cond = mu_x + rho * (sigma_x / sigma_y) * (current_point.y - mu_y);
                const std_dev_x_cond = sigma_x * Math.sqrt(1 - rho**2);
                const new_x = mean_x_cond + random_normal() * std_dev_x_cond;
                const intermediate_point = { x: new_x, y: current_point.y };
                fullHistory.push({ start: { ...current_point }, proposal: intermediate_point, accepted: true });

                const mean_y_cond = mu_y + rho * (sigma_y / sigma_x) * (new_x - mu_x);
                const std_dev_y_cond = sigma_y * Math.sqrt(1 - rho**2);
                const new_y = mean_y_cond + random_normal() * std_dev_y_cond;
                const new_point = { x: new_x, y: new_y };
                fullHistory.push({ start: intermediate_point, proposal: new_point, accepted: true });

                current_point = { ...new_point };
            }
            return fullHistory;
        }

        function runHMC(num_steps, L, epsilon) {
            let current_q = { ...start_point };
            const fullHistory = [];
            const U = (q) => -log_pdf_bivariate_normal(q.x, q.y);
            const K = (p) => 0.5 * (p.x**2 + p.y**2);

            for (let i = 0; i < num_steps; i++) {
                let q = { ...current_q };
                let p = { x: random_normal(), y: random_normal() };
                let current_p = { ...p };
                const trajectory = [{...q}];

                let grad = grad_log_pdf(q);
                p.x += 0.5 * epsilon * grad.x;
                p.y += 0.5 * epsilon * grad.y;

                for (let j = 0; j < L; j++) {
                    q.x += epsilon * p.x;
                    q.y += epsilon * p.y;
                    trajectory.push({...q});
                    if (j < L - 1) {
                        grad = grad_log_pdf(q);
                        p.x += epsilon * grad.x;
                        p.y += epsilon * grad.y;
                    }
                }
                
                grad = grad_log_pdf(q);
                p.x += 0.5 * epsilon * grad.x;
                p.y += 0.5 * epsilon * grad.y;

                const current_H = U(current_q) + K(current_p);
                const proposal_H = U(q) + K(p);
                const acceptance_ratio = Math.min(1, Math.exp(current_H - proposal_H));
                let accepted = (Math.random() < acceptance_ratio);

                fullHistory.push({ start: { ...current_q }, proposal: { ...q }, accepted: accepted, trajectory: trajectory });
                if (accepted) {
                    current_q = { ...q };
                }
            }
            return fullHistory;
        }

        // --- Plotting & Drawing ---
        
         // ADDED: New function to draw axis labels
        function drawAxesLabels() {
            ctx.fillStyle = '#333'; // Dark grey color for text
            ctx.font = 'bold 16px Arial';
            
            // X-axis label
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillText('X₁', canvasSize / 2, canvasSize - 5);
            
            // Y-axis label (requires rotation)
            ctx.save(); // Save the current state
            ctx.translate(15, canvasSize / 2); // Move origin to left-center
            ctx.rotate(-Math.PI / 2); // Rotate counter-clockwise
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillText('X₂', 5, 5);
            ctx.restore(); // Restore to the unrotated state
        }
        
        function toCanvasCoords(point) {
            const scale = canvasSize / (domain.max - domain.min);
            return { x: (point.x - domain.min) * scale, y: canvasSize - (point.y - domain.min) * scale };
        }

        function drawDistributionContours() {
            const center = toCanvasCoords(targetMean);
            const scale = canvasSize / (domain.max - domain.min);
            const varX = targetStdDev.x ** 2; 
            const varY = targetStdDev.y ** 2; 
            const covXY = targetCorrelation * targetStdDev.x * targetStdDev.y;
            const term1 = varX + varY; 
            const term2 = Math.sqrt((varX - varY) ** 2 + 4 * covXY ** 2);
            const eigenvalue1 = (term1 + term2) / 2; 
            const eigenvalue2 = (term1 - term2) / 2;
            const semiAxis1 = Math.sqrt(eigenvalue1); 
            const semiAxis2 = Math.sqrt(eigenvalue2);
            const angle = 0.5 * Math.atan2(2 * covXY, varX - varY);
            for (let i = 3; i > 0; i--) {
                ctx.beginPath();
                ctx.fillStyle = `rgba(173, 216, 230, ${0.12 * (4-i)})`;
                ctx.ellipse(center.x, center.y, i * semiAxis1 * scale, i * semiAxis2 * scale, -angle, 0, 2 * Math.PI);
                ctx.fill();
            }
        }

        function drawBackground() {
            ctx.clearRect(0, 0, canvasSize, canvasSize);
            ctx.strokeStyle = '#e0e0e0';
            ctx.lineWidth = 0.5;
            for (let i = Math.ceil(domain.min); i <= Math.floor(domain.max); i++) {
                const p = toCanvasCoords({ x: i, y: i });
                ctx.beginPath(); 
                ctx.moveTo(p.x, 0); 
                ctx.lineTo(p.x, canvasSize); 
                ctx.stroke();
                ctx.beginPath(); 
                ctx.moveTo(0, p.y); 
                ctx.lineTo(canvasSize, p.y); 
                ctx.stroke();
            }
            drawDistributionContours();
            drawAxesLabels();
        }

        function drawArrow(from, to, color, lineWidth = 1.5) {
            const headlen = 8;
            const canvasFrom = toCanvasCoords(from); 
            const canvasTo = toCanvasCoords(to);
            const dx = canvasTo.x - canvasFrom.x; 
            const dy = canvasTo.y - canvasFrom.y;
            const angle = Math.atan2(dy, dx);
            ctx.strokeStyle = color; 
            ctx.fillStyle = color; 
            ctx.lineWidth = lineWidth;
            ctx.beginPath(); 
            ctx.moveTo(canvasFrom.x, canvasFrom.y); 
            ctx.lineTo(canvasTo.x, canvasTo.y); 
            ctx.stroke();
            ctx.beginPath(); 
            ctx.moveTo(canvasTo.x, canvasTo.y);
            ctx.lineTo(canvasTo.x - headlen * Math.cos(angle - Math.PI / 6), canvasTo.y - headlen * Math.sin(angle - Math.PI / 6));
            ctx.lineTo(canvasTo.x - headlen * Math.cos(angle + Math.PI / 6), canvasTo.y - headlen * Math.sin(angle + Math.PI / 6));
            ctx.closePath(); 
            ctx.fill();
        }

        function drawPoint(point, color, size = 3) {
            const canvasPoint = toCanvasCoords(point);
            ctx.beginPath();
            ctx.arc(canvasPoint.x, canvasPoint.y, size, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
        }

        function drawTrajectory(path, color) {
            if (path.length < 2) return;
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            const start = toCanvasCoords(path[0]);
            ctx.moveTo(start.x, start.y);
            for (let i = 1; i < path.length; i++) {
                const p = toCanvasCoords(path[i]);
                ctx.lineTo(p.x, p.y);
            }
            ctx.stroke();
        }
        
      function drawMarginalCurve(mCtx, mean, stdDev, yMaxCount, numCurrentSamples) {
            if (yMaxCount === 0 || numCurrentSamples === 0) return;

            const width = mCtx.canvas.width;
            const height = mCtx.canvas.height;
            
            // Use the exact same scaling factor as the histogram.
            const scaleY = (height - 5) / yMaxCount;
            const scaleX = width / (domain.max - domain.min);
            
            mCtx.beginPath();
            mCtx.strokeStyle = 'rgba(30, 144, 255, 0.8)';
            mCtx.lineWidth = 2;

            for (let px = 0; px < width; px++) {
                const x = px / scaleX + domain.min;
                // Calculate the expected count for this x-value.
                const theoreticalCount = numCurrentSamples * pdf_normal(x, mean, stdDev) * binWidth;
                // Scale the count to the pixel grid.
                const y = height - theoreticalCount * scaleY;
                
                if (px === 0) mCtx.moveTo(px, y);
                else mCtx.lineTo(px, y);
            }
            mCtx.stroke();
        }
        
        function drawHistogram(mCtx, data, yMaxCount) {
            if (data.length === 0 || yMaxCount === 0) return;
            const width = mCtx.canvas.width;
            const height = mCtx.canvas.height;
            const bins = new Array(numBins).fill(0);
            data.forEach(d => {
                if (d >= domain.min && d < domain.max) {
                    const binIndex = Math.floor((d - domain.min) / binWidth);
                    bins[binIndex]++;
                }
            });
            const scaleY = (height - 5) / yMaxCount; // padding
            const scaleX = width / numBins;
            mCtx.fillStyle = 'rgba(255, 165, 0, 0.6)';
            bins.forEach((count, i) => {
                const barHeight = count * scaleY;
                mCtx.fillRect(i * scaleX, height - barHeight, scaleX - 1, barHeight);
            });
        }
        
        // --- Main Animation Loop ---
        function animate(timestamp) {
            if (!isPaused && (timestamp - lastUpdateTime > stepDelay)) {
                lastUpdateTime = timestamp;
                if (currentStep < n_animation_steps) {
                    currentStep++;
                }
            }
            drawBackground();
            
            const currentPath = chainPath.slice(0, currentStep + 1);
            for (let i = 0; i < currentPath.length; i++) {
                drawPoint(currentPath[i], 'rgba(0, 0, 0, 0.6)');
            }

            if (currentStep > 0 && currentStep <= history.length) {
                const latestStep = history[currentStep - 1];
                if (latestStep.trajectory) {
                    drawTrajectory(latestStep.trajectory, 'rgba(128, 128, 128, 0.6)');
                }
                const color = latestStep.accepted ? 'rgba(30, 144, 255, 0.7)' : 'rgba(255, 0, 0, 0.45)';
                drawArrow(latestStep.start, latestStep.proposal, color, 2);
            }
            drawPoint(start_point, 'green', 5);

            // Marginal Plots
            const acceptedX = currentPath.map(p => p.x);
            const acceptedY = currentPath.map(p => p.y);
            marginalXCtx.clearRect(0, 0, marginalXCanvas.width, marginalXCanvas.height);
            drawHistogram(marginalXCtx, acceptedX, yMaxCountX);
            drawMarginalCurve(marginalXCtx, targetMean.x, targetStdDev.x, yMaxCountX, acceptedX.length);
            marginalYCtx.clearRect(0, 0, marginalYCanvas.width, marginalYCanvas.height);
            drawHistogram(marginalYCtx, acceptedY, yMaxCountY);
            drawMarginalCurve(marginalYCtx, targetMean.y, targetStdDev.y, yMaxCountY, acceptedY.length);
            
            if (currentStep >= n_animation_steps && !isPaused) {
                cancelAnimationFrame(animationFrameId);
                return;
            }
            animationFrameId = requestAnimationFrame(animate);
        }
        
        // --- Control & Setup ---
        function updateUIControls() {
            const algorithm = algorithmSelector.value;
            const display = (algorithm === 'hmc') ? 'flex' : 'none';
            hmcControls1.style.display = display;
            hmcControls2.style.display = display;
        }

        function startAnimation() {
            if (animationFrameId) cancelAnimationFrame(animationFrameId);
            
            isPaused = false;
            pauseButton.textContent = 'Pause';
            
            const numSamples = parseInt(numPointsInput.value, 10);
            const algorithm = algorithmSelector.value;
            updateUIControls();
            
            if (algorithm === 'mh') {
                descriptionP.innerHTML = `This animation visualizes the <strong>Metropolis-Hastings</strong> algorithm drawing points from a Bivariate Normal distribution.`;
                animationKeyUl.innerHTML = `
                  <li><strong>Shawdowed Elipses:</strong> Distribution to be recovered - a bivariate normal distribution.</li>
                  <li><strong>Arrow Red:</strong> Show a proposed jump that was rejectedand no point is sampled.</li>
                  <li><strong>Arrow Blue:</strong> Show a proposed jump that was accepted and the point is sampled.</li>
                  <li><strong>Dots:</strong> Drawn points.</li>
                  <li><strong>Histograms:</strong> Marginal distributions of X<sub>1</sub>and X<sub>2</sub>.</li>`;
                history = runMCMC(numSamples);
            } else if (algorithm === 'gibbs') {
                descriptionP.innerHTML = `This animation visualizes <strong>Gibbs Sampling</strong> algorithm drawing points from a Bivariate Normal distribution. Each draw requires two steps: a draw from p(x<sub>1</sub>|x<sub>2</sub>) then a draw from p(x<sub>2</sub>|x<sub>1</sub>).`;
                animationKeyUl.innerHTML = `
                    <li><strong>Shawdowed Elipses:</strong> Distribution to be recovered - a bivariate normal distribution.</li>
                    <li><strong>Arrow Blue:</strong> A conditional draw (always accepted). Horizontal arrows sample X<sub>1</sub>, vertical arrows sample  X<sub>2</sub>.</li>
                    <li><strong>Dots:</strong> Drawn points.</li>
                    <li><strong>Histograms:</strong> Marginal distributions of X<sub>1</sub> and X<sub>2</sub>.</li>`;
                history = runGibbs(numSamples);
            } else { // hmc
                descriptionP.innerHTML = `This animation visualizes <strong>Hamiltonian Monte Carlo</strong>. It uses a physics simulation (leapfrog trajectory) to generate smart proposals.`;
                animationKeyUl.innerHTML = `
                  <li><strong>Shawdowed Elipses:</strong> Distribution to be recovered - a bivariate normal distribution.</li>
                  <li><strong>Grey Path:</strong> The leapfrog trajectory used to find a proposal.</li>
                  <li><strong>Arrow Red:</strong> Show a proposed jump that was rejectedand no point is sampled.</li>
                  <li><strong>Arrow Blue:</strong> Show a proposed jump that was accepted and the point is sampled.</li>
                  <li><strong>Dots:</strong> Drawn points.</li>
                  <li><strong>Histograms:</strong> Marginal distributions of X<sub>1</sub> and X<sub>2</sub>.</li>`;
                const L = parseInt(leapfrogStepsInput.value, 10);
                const epsilon = parseFloat(epsilonInput.value);
                history = runHMC(numSamples, L, epsilon);
            }
            
            n_animation_steps = history.length;
            
            chainPath = [start_point];
            let lastPoint = start_point;
            for (const step of history) {
                lastPoint = step.accepted ? step.proposal : lastPoint;
                chainPath.push(lastPoint);
            }

            const expectedMaxCountX = n_animation_steps * pdf_normal(targetMean.x, targetMean.x, targetStdDev.x) * binWidth;
            yMaxCountX = expectedMaxCountX * 1.5;
            const expectedMaxCountY = n_animation_steps * pdf_normal(targetMean.y, targetMean.y, targetStdDev.y) * binWidth;
            yMaxCountY = expectedMaxCountY * 1.5;

            currentStep = 0;
            lastUpdateTime = 0;
            animationFrameId = requestAnimationFrame(animate);
        }

        // --- Event Listeners ---
        speedSlider.addEventListener('input', () => {
            stepDelay = parseInt(speedSlider.value, 10);
            speedValueSpan.textContent = `${stepDelay} ms`;
        });
        
        pauseButton.addEventListener('click', () => {
            isPaused = !isPaused;
            if (isPaused) {
                pauseButton.textContent = 'Resume';
                cancelAnimationFrame(animationFrameId); // Stop animation loop
            } else {
                pauseButton.textContent = 'Pause';
                lastUpdateTime = performance.now(); // Reset timer to prevent jump
                animationFrameId = requestAnimationFrame(animate); // Resume animation
            }
        });
        
        algorithmSelector.addEventListener('change', startAnimation);
        restartButton.addEventListener('click', startAnimation);
        
        window.onload = () => {
            speedSlider.dispatchEvent(new Event('input'));
            startAnimation();
        };
    
